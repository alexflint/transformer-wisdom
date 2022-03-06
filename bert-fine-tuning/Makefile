TPU := bert-fine-tuning		# TPU name
TYPE := v2-8  				# TPU type (just v2-8 or v3-8)
ZONE := us-central1-b       # Must be one of us-central1-[abc], europe-west4-a, or asia-east1-c
TENSORFLOW_VERSION := 2.8.0

create-tpu:
	gcloud compute tpus create $(TPU) \
		--zone $(ZONE) \
		--version $(TENSORFLOW_VERSION) \
		--accelerator-type $(TYPE)

delete-tpu:
	gcloud compute tpus delete $(TPU) \
		--zone $(ZONE)

describe-tpu:
	gcloud compute tpus describe $(TPU) \
		--zone=$(ZONE)



EG := my-execution-group

create-eg:
	gcloud compute tpus execution-groups create \
		--name $(EG) \
		--zone $(ZONE) \
		--tf-version $(TENSORFLOW_VERSION) \
		--accelerator-type $(TYPE)

delete-eg:
	gcloud compute tpus execution-groups delete $(EG) \
		--zone $(ZONE)

ssh:
	gcloud compute ssh $(EG) \
		--zone $(ZONE)

# create firewall ruleset

# gcloud compute firewall-rules create $(RULESET_NAME) \
#    --network=NETWORK \
#    --allow=tcp:22

# use identity-aware proxy to tunnel to a raw TPU?
# https://cloud.google.com/iap/docs/tcp-forwarding-overview