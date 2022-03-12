TPU := bert-fine-tuning		# TPU name
TYPE := v3-8  				# TPU type (just v2-8 or v3-8)
TENSORFLOW_RUNTIME := 2.8.0
NETWORK := default

ZONE := $(shell cat Makefile.zone)   # Must be one of us-central1-[abc], europe-west4-a, or asia-east1-c

# On google cloud there are TPUs, TPU VMs, and TPU execution groups. A TPU
# means that a physical TPU is allocated and we have access to it via a gRPC
# endpoint. A TPU VM means that additionally we get SSH access to the worker
# that is physically attached to the TPU. A TPU execution group means that we
# get a TPU plus a separate compute instance that is set up with network
# access to the TPU.
#
# Here we only really want a TPU, but a TPU never gets an external IP address,
# so in order to access it we would need to set up a separate bastion compute
# instance, which is an operational pain. Therefore we allocate a TPU VM just
# so that we get an external IP address.
#
# On the other hand raw TPUs seem to be more readily available than TPU VMs,
# based on my experience trying to create both.

create-tpu:
	gcloud compute tpus create $(TPU) \
		--zone $(ZONE) \
		--version $(TENSORFLOW_RUNTIME) \
		--network $(NETWORK) \
		--accelerator-type $(TYPE)

delete-tpu:
	gcloud compute tpus delete $(TPU) \
		--zone $(ZONE)

describe-tpu:
	gcloud compute tpus describe $(TPU) \
		--zone=$(ZONE)

list-tensorflow-versions:
	gcloud alpha compute tpus tpu-vm versions list \
		--zone=$(ZONE)

BASTION := bastion

create-bastion:
	gcloud compute instances create $(BASTION) \
		--zone $(ZONE) \
		--machine-type e2-micro \
		--network $(NETWORK)

delete-bastion:
	gcloud compute instances delete $(BASTION)

# create a tunnel to the TPU worker grpc endpoint
tunnel:
	gcloud compute ssh $(BASTION) \
		--zone=$(ZONE) \
		-- \
		-N \
		-L 19870:$(shell \
			gcloud compute tpus describe bert-fine-tuning \
				--zone=$(ZONE) \
				--format='get(ipAddress,port)[separator=:]')



#
# OLD: TPU VMs

TPU_RUNTIME := tpu-vm-tf-$(TENSORFLOW_RUNTIME)

create-tpu-vm:
	gcloud alpha compute tpus tpu-vm create $(TPU) \
		--zone $(ZONE) \
		--version $(TPU_RUNTIME) \
		--network $(NETWORK) \
		--accelerator-type $(TYPE)

#
# OLD: TPU execution groups
#

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