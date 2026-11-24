output "api_url" {
  value = "${aws_apigatewayv2_api.http.api_endpoint}/v1/query"
}

output "docs_bucket" {
  description = "Upload documents as s3://<bucket>/<tenant>/<path>."
  value       = aws_s3_bucket.docs.bucket
}

output "user_pool_id" {
  value = aws_cognito_user_pool.users.id
}

output "app_client_id" {
  value = aws_cognito_user_pool_client.app.id
}

output "reindex_state_machine" {
  value = aws_sfn_state_machine.reindex.arn
}
