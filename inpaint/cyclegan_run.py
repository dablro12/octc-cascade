#!/usr/bin/env python
from UNET_train import Train 
from utils.arg import get_args 


def main():
    args = get_args()
    trainer =  Train(args = args)
    trainer.fit()
    
if __name__ == '__main__':
    main() 