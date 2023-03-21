# Distributed-Training

Running ResNet50 on ImageNet dataset in case:
- Single GPU (baseline)
- Data Parallel 2 GPUs
- Distributed Data Parallism 2 GPUs
- Distributed Data Parallism 2 GPU + Mixed Precision 



# Running
>>python train_1gpu.py

>>python train_dp.py

>>torchrun --standalone --nproc_per_node=2 train_ddp.py

>>torchrun --standalone --nproc_per_node=2 train_ddp_mixed_precision.py


# NOTE
## Torchrun
Torchrun handles the minutiae of distributed training itself:
- Users do not need to set environment variables or explicitly switch between rank and world_size; torchrun specifies this along with several other environment variables.
- Users do not need to call mp.spawn in scripts, just a main() function and initialize the script with torchrun. This way the same script can be run with a non-distributed installation as well as single-node or multinode.
- If an error occurs, torchrun will terminate all processes and restart training from the last saved training snapshot.
- In elastic training, whenever nodes are added/removed, torchrun will end and spawn the process on available devices without manual intervention.

## Mixed Precision
- Mixed precision is a combination that uses both 16-bit precision and 32-bit precision in training.
- Goal: Cuda's 16-bit precision support reduces the storage capacity of the model in half, allows for larger batchsize processing, and increases computational speed while maintaining model performance and accuracy. .
- Normally, Pytorch defaults to 32-bit precision tensors for loss and weights calculations. This causes the computer to use more RAM for storage, sometimes causing us to reduce batchsize or choose models with less parameters to be able to train.
