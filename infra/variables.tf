variable "name" {
  type    = string
  default = "srag"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "embed_model_id" {
  type    = string
  default = "amazon.titan-embed-text-v2:0"
}

variable "gen_model_id" {
  description = "Claude model ID or inference-profile ID as listed in your Bedrock console."
  type        = string
}

variable "gen_model_match" {
  description = "Substring of the generation model ID used to scope bedrock:InvokeModel (e.g. \"claude\")."
  type        = string
  default     = "anthropic.claude"
}

variable "min_score" {
  description = "Abstain below this retrieval score. Embedder-specific: calibrate with `python -m srag.evaluate` against your embedder."
  type        = number
  default     = 0.2
}

variable "usd_per_mtok_in" {
  type    = number
  default = 0
}

variable "usd_per_mtok_out" {
  type    = number
  default = 0
}

variable "ingest_concurrency" {
  type    = number
  default = 5
}

variable "api_rate" {
  type    = number
  default = 20
}

variable "api_burst" {
  type    = number
  default = 40
}
