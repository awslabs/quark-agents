#!/bin/bash
# setup_iam.sh — one-time IAM setup for Quark Ray cluster
#
# Creates a dedicated IAM role with exactly the permissions needed:
#   - EC2: launch/terminate/describe instances (for Ray autoscaler)
#   - Bedrock: invoke models (for agent LLM calls)
#   - SSM: session manager access (for secure node access)
#   - IAM: PassRole (autoscaler must pass this role to new worker instances)
#
# Usage: bash benchmarks/ray-cluster/setup_iam.sh
# Safe to re-run (idempotent).

set -e

ROLE_NAME="QuarkRayRole"
PROFILE_NAME="QuarkRayInstanceProfile"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_DEFAULT_REGION:-us-west-2}

echo "Setting up IAM role: $ROLE_NAME (account: $ACCOUNT_ID, region: $REGION)"

if aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
    echo "Role $ROLE_NAME already exists, skipping creation."
else
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' \
        --description "Role for Quark Ray cluster nodes"
    echo "Created role: $ROLE_NAME"
fi

for policy in \
    "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore" \
    "arn:aws:iam::aws:policy/AmazonEC2FullAccess" \
    "arn:aws:iam::aws:policy/AmazonBedrockFullAccess"; do
    aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$policy" 2>/dev/null || true
    echo "Attached: $policy"
done

# iam:PassRole is required so the autoscaler can assign this role to new worker instances
aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "QuarkRayPassRole" \
    --policy-document "{
        \"Version\": \"2012-10-17\",
        \"Statement\": [{
            \"Effect\": \"Allow\",
            \"Action\": \"iam:PassRole\",
            \"Resource\": \"arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}\"
        }]
    }"
echo "Added inline PassRole policy"

if aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" &>/dev/null; then
    echo "Instance profile $PROFILE_NAME already exists, skipping."
else
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME"
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$PROFILE_NAME" \
        --role-name "$ROLE_NAME"
    echo "Created instance profile: $PROFILE_NAME"
fi

echo ""
echo "IAM setup complete."
echo "  Role ARN    : arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo "  Profile Name: $PROFILE_NAME"
echo ""
echo "Next: ray up benchmarks/ray-cluster/ray-cluster.yaml --yes"
