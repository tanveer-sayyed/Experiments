from datetime import datetime
import torchvision
import torchvision.transforms as transforms
import torch
import torch.distributed as dist
import torch.nn as nn


from model import ConvNet

def train(gpu, args):
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(
    	backend='nccl', # NVIDIA Collective Communication Library
    	init_method='env://',
    	world_size=args.world_size,
    	rank=rank 
    )

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        	train_dataset,
        	num_replicas=args.world_size,
        	rank=rank
        )
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler
        )
    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 5 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))

    if rank == 0:
            dict_model = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': args.epochs,
            }
            torch.save(dict_model, './model.pth')
