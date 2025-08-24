terraform {
  required_version = ">= 1.5"
  backend "s3" {
    bucket = "cs2bet-tf-state"
    key    = "global/terraform.tfstate"
    region = "ap-southeast-1"
  }
}
provider "aws" {
  region = "ap-southeast-1"
}
