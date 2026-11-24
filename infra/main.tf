terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { app = var.name, managed-by = "terraform" }
  }
}

data "aws_caller_identity" "me" {}
data "aws_partition" "p" {}

locals {
  account = data.aws_caller_identity.me.account_id
  embed_model_arn = "arn:${data.aws_partition.p.partition}:bedrock:${var.region}::foundation-model/${var.embed_model_id}"
  # Generation may go through a cross-region inference profile; both ARNs are allowed.
  gen_arns = [
    "arn:${data.aws_partition.p.partition}:bedrock:*::foundation-model/*${var.gen_model_match}*",
    "arn:${data.aws_partition.p.partition}:bedrock:${var.region}:${local.account}:inference-profile/*${var.gen_model_match}*",
  ]
  lambda_env = {
    LEDGER_TABLE     = aws_dynamodb_table.ledger.name
    SESSIONS_TABLE   = aws_dynamodb_table.sessions.name
    AOSS_ENDPOINT    = aws_opensearchserverless_collection.vectors.collection_endpoint
    AOSS_INDEX       = "chunks"
    EMBED_MODEL_ID   = var.embed_model_id
    GEN_MODEL_ID     = var.gen_model_id
    TOP_K            = "5"
    MIN_SCORE        = tostring(var.min_score)
    USD_PER_MTOK_IN  = tostring(var.usd_per_mtok_in)
    USD_PER_MTOK_OUT = tostring(var.usd_per_mtok_out)
  }
}

# --- Storage ------------------------------------------------------------------

resource "aws_kms_key" "data" {
  description         = "${var.name} documents, tables and logs"
  enable_key_rotation = true
}

resource "aws_s3_bucket" "docs" {
  bucket_prefix = "${var.name}-docs-"
}

resource "aws_s3_bucket_public_access_block" "docs" {
  bucket                  = aws_s3_bucket.docs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "docs" {
  bucket = aws_s3_bucket.docs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "docs" {
  bucket = aws_s3_bucket.docs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.data.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_dynamodb_table" "ledger" {
  name         = "${var.name}-ledger"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  attribute {
    name = "pk"
    type = "S"
  }
  point_in_time_recovery {
    enabled = true
  }
  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.data.arn
  }
}

resource "aws_dynamodb_table" "sessions" {
  name         = "${var.name}-sessions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "ts"
  attribute {
    name = "pk"
    type = "S"
  }
  attribute {
    name = "ts"
    type = "N"
  }
  ttl {
    attribute_name = "expires"
    enabled        = true
  }
  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.data.arn
  }
}

# --- OpenSearch Serverless (vector search) -----------------------------------

resource "aws_opensearchserverless_security_policy" "enc" {
  name = "${var.name}-enc"
  type = "encryption"
  policy = jsonencode({
    Rules       = [{ ResourceType = "collection", Resource = ["collection/${var.name}"] }]
    AWSOwnedKey = true
  })
}

resource "aws_opensearchserverless_security_policy" "net" {
  name = "${var.name}-net"
  type = "network"
  policy = jsonencode([{
    Rules = [
      { ResourceType = "collection", Resource = ["collection/${var.name}"] },
      { ResourceType = "dashboard", Resource = ["collection/${var.name}"] },
    ]
    AllowFromPublic = true # SigV4 + data-access policy still required; use a VPC endpoint in prod
  }])
}

resource "aws_opensearchserverless_collection" "vectors" {
  name       = var.name
  type       = "VECTORSEARCH"
  depends_on = [aws_opensearchserverless_security_policy.enc, aws_opensearchserverless_security_policy.net]
}

resource "aws_opensearchserverless_access_policy" "data" {
  name = "${var.name}-data"
  type = "data"
  policy = jsonencode([{
    Rules = [
      {
        ResourceType = "index"
        Resource     = ["index/${var.name}/*"]
        Permission   = ["aoss:CreateIndex", "aoss:DescribeIndex", "aoss:ReadDocument", "aoss:WriteDocument", "aoss:UpdateIndex"]
      },
    ]
    Principal = [aws_iam_role.ingest.arn, aws_iam_role.query.arn]
  }])
}

# --- Lambda packaging ----------------------------------------------------------
# `make package` vendors dependencies into build/lambda before `terraform apply`.

data "archive_file" "lambda" {
  type        = "zip"
  source_dir  = "${path.module}/../build/lambda"
  output_path = "${path.module}/../build/lambda.zip"
}

# --- IAM (one role per function, least privilege) -----------------------------

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ingest" {
  name               = "${var.name}-ingest"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role" "query" {
  name               = "${var.name}-query"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "logs" {
  for_each   = { ingest = aws_iam_role.ingest.name, query = aws_iam_role.query.name }
  role       = each.value
  policy_arn = "arn:${data.aws_partition.p.partition}:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "ingest" {
  statement {
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.docs.arn, "${aws_s3_bucket.docs.arn}/*"]
  }
  statement {
    actions   = ["dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:GetItem"]
    resources = [aws_dynamodb_table.ledger.arn]
  }
  statement {
    actions   = ["bedrock:InvokeModel"]
    resources = [local.embed_model_arn]
  }
  statement {
    actions   = ["aoss:APIAccessAll"]
    resources = [aws_opensearchserverless_collection.vectors.arn]
  }
  statement {
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = [aws_kms_key.data.arn]
  }
  statement {
    actions   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"]
    resources = [aws_sqs_queue.ingest.arn]
  }
}

resource "aws_iam_role_policy" "ingest" {
  role   = aws_iam_role.ingest.id
  policy = data.aws_iam_policy_document.ingest.json
}

data "aws_iam_policy_document" "query" {
  statement {
    actions   = ["dynamodb:PutItem"]
    resources = [aws_dynamodb_table.sessions.arn]
  }
  statement {
    actions   = ["bedrock:InvokeModel"]
    resources = concat([local.embed_model_arn], local.gen_arns)
  }
  statement {
    actions   = ["aoss:APIAccessAll"]
    resources = [aws_opensearchserverless_collection.vectors.arn]
  }
  statement {
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = [aws_kms_key.data.arn]
  }
}

resource "aws_iam_role_policy" "query" {
  role   = aws_iam_role.query.id
  policy = data.aws_iam_policy_document.query.json
}

# --- Ingest path: S3 -> SQS (buffer + DLQ) -> Lambda ---------------------------

resource "aws_sqs_queue" "ingest_dlq" {
  name                      = "${var.name}-ingest-dlq"
  message_retention_seconds = 1209600
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "ingest" {
  name                       = "${var.name}-ingest"
  visibility_timeout_seconds = 6 * 120 # >= 6x the function timeout (AWS guidance)
  sqs_managed_sse_enabled    = true
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.ingest_dlq.arn
    maxReceiveCount     = 3
  })
}

data "aws_iam_policy_document" "s3_to_sqs" {
  statement {
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.ingest.arn]
    principals {
      type        = "Service"
      identifiers = ["s3.amazonaws.com"]
    }
    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values   = [aws_s3_bucket.docs.arn]
    }
  }
}

resource "aws_sqs_queue_policy" "ingest" {
  queue_url = aws_sqs_queue.ingest.id
  policy    = data.aws_iam_policy_document.s3_to_sqs.json
}

resource "aws_s3_bucket_notification" "docs" {
  bucket = aws_s3_bucket.docs.id
  queue {
    queue_arn = aws_sqs_queue.ingest.arn
    events    = ["s3:ObjectCreated:*", "s3:ObjectRemoved:*"]
  }
  depends_on = [aws_sqs_queue_policy.ingest]
}

resource "aws_lambda_function" "ingest" {
  function_name    = "${var.name}-ingest"
  role             = aws_iam_role.ingest.arn
  runtime          = "python3.12"
  architectures    = ["arm64"]
  handler          = "srag.handlers.ingest_handler"
  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256
  timeout          = 120
  memory_size      = 1024
  # Caps concurrent Bedrock embedding calls so a bulk upload can't exhaust the account quota.
  reserved_concurrent_executions = var.ingest_concurrency
  environment {
    variables = local.lambda_env
  }
  tracing_config {
    mode = "Active"
  }
}

resource "aws_lambda_event_source_mapping" "ingest" {
  event_source_arn                   = aws_sqs_queue.ingest.arn
  function_name                      = aws_lambda_function.ingest.arn
  batch_size                         = 10
  maximum_batching_window_in_seconds = 5
  function_response_types            = ["ReportBatchItemFailures"]
}

# --- Query path: API Gateway (JWT) -> Lambda ----------------------------------

resource "aws_lambda_function" "query" {
  function_name    = "${var.name}-query"
  role             = aws_iam_role.query.arn
  runtime          = "python3.12"
  architectures    = ["arm64"]
  handler          = "srag.handlers.query_handler"
  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256
  timeout          = 60
  memory_size      = 1024
  environment {
    variables = local.lambda_env
  }
  tracing_config {
    mode = "Active"
  }
}

resource "aws_cognito_user_pool" "users" {
  name = "${var.name}-users"
  schema {
    name                = "tenant"
    attribute_data_type = "String"
    mutable             = false # a user can never move themselves to another tenant
    string_attribute_constraints {
      min_length = 1
      max_length = 64
    }
  }
  password_policy {
    minimum_length    = 12
    require_symbols   = true
    require_numbers   = true
    require_uppercase = true
    require_lowercase = true
  }
}

resource "aws_cognito_user_pool_client" "app" {
  name            = "${var.name}-app"
  user_pool_id    = aws_cognito_user_pool.users.id
  read_attributes = ["custom:tenant", "email"]
  # Users must not be able to write their own tenant attribute.
  write_attributes    = ["email"]
  explicit_auth_flows = ["ALLOW_USER_SRP_AUTH", "ALLOW_REFRESH_TOKEN_AUTH"]
}

resource "aws_apigatewayv2_api" "http" {
  name          = var.name
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_authorizer" "jwt" {
  api_id           = aws_apigatewayv2_api.http.id
  authorizer_type  = "JWT"
  identity_sources = ["$request.header.Authorization"]
  name             = "cognito"
  jwt_configuration {
    audience = [aws_cognito_user_pool_client.app.id]
    issuer   = "https://cognito-idp.${var.region}.amazonaws.com/${aws_cognito_user_pool.users.id}"
  }
}

resource "aws_apigatewayv2_integration" "query" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.query.invoke_arn
  payload_format_version = "2.0"
  timeout_milliseconds   = 30000
}

resource "aws_apigatewayv2_route" "query" {
  api_id             = aws_apigatewayv2_api.http.id
  route_key          = "POST /v1/query"
  target             = "integrations/${aws_apigatewayv2_integration.query.id}"
  authorization_type = "JWT"
  authorizer_id      = aws_apigatewayv2_authorizer.jwt.id
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true
  default_route_settings {
    throttling_burst_limit = var.api_burst
    throttling_rate_limit  = var.api_rate
  }
}

resource "aws_lambda_permission" "api" {
  statement_id  = "api"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.query.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*/v1/query"
}

# --- Bulk re-index: Step Functions (list -> Map -> ingest) -------------------

resource "aws_lambda_function" "list" {
  function_name    = "${var.name}-reindex-list"
  role             = aws_iam_role.ingest.arn
  runtime          = "python3.12"
  architectures    = ["arm64"]
  handler          = "srag.handlers.list_for_reindex_handler"
  filename         = data.archive_file.lambda.output_path
  source_code_hash = data.archive_file.lambda.output_base64sha256
  timeout          = 60
  environment {
    variables = local.lambda_env
  }
}

data "aws_iam_policy_document" "sfn_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sfn" {
  name               = "${var.name}-reindex"
  assume_role_policy = data.aws_iam_policy_document.sfn_assume.json
}

resource "aws_iam_role_policy" "sfn" {
  role = aws_iam_role.sfn.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["lambda:InvokeFunction"]
      Resource = [aws_lambda_function.list.arn, aws_lambda_function.ingest.arn]
    }]
  })
}

resource "aws_sfn_state_machine" "reindex" {
  name     = "${var.name}-reindex"
  role_arn = aws_iam_role.sfn.arn
  definition = jsonencode({
    Comment = "Re-embed every document under a prefix (e.g. after changing the embedding model)"
    StartAt = "List"
    States = {
      List = {
        Type       = "Task"
        Resource   = "arn:${data.aws_partition.p.partition}:states:::lambda:invoke"
        Parameters = { FunctionName = aws_lambda_function.list.arn, "Payload.$" = "$" }
        ResultSelector = { "batches.$" = "$.Payload.batches", "count.$" = "$.Payload.count" }
        Next = "Ingest"
      }
      Ingest = {
        Type           = "Map"
        ItemsPath      = "$.batches"
        MaxConcurrency = var.ingest_concurrency
        ItemProcessor = {
          ProcessorConfig = { Mode = "INLINE" }
          StartAt         = "IngestBatch"
          States = {
            IngestBatch = {
              Type       = "Task"
              Resource   = "arn:${data.aws_partition.p.partition}:states:::lambda:invoke"
              Parameters = { FunctionName = aws_lambda_function.ingest.arn, "Payload.$" = "$" }
              Retry = [{
                ErrorEquals     = ["Lambda.TooManyRequestsException", "Lambda.ServiceException", "States.Timeout"]
                IntervalSeconds = 2
                BackoffRate     = 2
                MaxAttempts     = 5
                JitterStrategy  = "FULL"
              }]
              End = true
            }
          }
        }
        End = true
      }
    }
  })
}

# --- Alarms --------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "dlq" {
  alarm_name          = "${var.name}-ingest-dlq"
  alarm_description   = "Documents failed ingestion 3 times and are parked in the DLQ."
  namespace           = "AWS/SQS"
  metric_name         = "ApproximateNumberOfMessagesVisible"
  dimensions          = { QueueName = aws_sqs_queue.ingest_dlq.name }
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
}

resource "aws_cloudwatch_metric_alarm" "abstain_rate" {
  alarm_name          = "${var.name}-abstain-spike"
  alarm_description   = "More than 40% of answers abstain: retrieval or index is likely broken."
  namespace           = "ServerlessRAG"
  metric_name         = "Abstained"
  dimensions          = { Function = "query" }
  statistic           = "Average"
  period              = 900
  evaluation_periods  = 2
  threshold           = 0.4
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
}

resource "aws_cloudwatch_metric_alarm" "query_errors" {
  alarm_name          = "${var.name}-query-errors"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.query.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
}
