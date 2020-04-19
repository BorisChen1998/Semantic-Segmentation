import torch
class Config(object):
    def __init__(self, dataset, only_encode, extra_data):
        self.NAME= "myNet"

        self.dataset = dataset
        self.only_encode = only_encode
        self.extra_data = extra_data
        
        if dataset == "cityscapes":
            self.weight = torch.ones(20)
            if only_encode:
                self.weight[0] = 2.3653597831726	
                self.weight[1] = 4.4237880706787	
                self.weight[2] = 2.9691488742828	
                self.weight[3] = 5.3442072868347	
                self.weight[4] = 5.2983593940735	
                self.weight[5] = 5.2275490760803	
                self.weight[6] = 5.4394111633301	
                self.weight[7] = 5.3659925460815	
                self.weight[8] = 3.4170460700989	
                self.weight[9] = 5.2414722442627	
                self.weight[10] = 4.7376127243042	
                self.weight[11] = 5.2286224365234	
                self.weight[12] = 5.455126285553	
                self.weight[13] = 4.3019247055054	
                self.weight[14] = 5.4264230728149	
                self.weight[15] = 5.4331531524658	
                self.weight[16] = 5.433765411377	
                self.weight[17] = 5.4631009101868	
                self.weight[18] = 5.3947434425354   
            elif not extra_data:
                self.weight[0] = 2.8149201869965	
                self.weight[1] = 6.9850029945374	
                self.weight[2] = 3.7890393733978	
                self.weight[3] = 9.9428062438965	
                self.weight[4] = 9.7702074050903	
                self.weight[5] = 9.5110931396484	
                self.weight[6] = 10.311357498169	
                self.weight[7] = 10.026463508606	
                self.weight[8] = 4.6323022842407	
                self.weight[9] = 9.5608062744141	
                self.weight[10] = 7.8698215484619	
                self.weight[11] = 9.5168733596802	
                self.weight[12] = 10.373730659485	
                self.weight[13] = 6.6616044044495	
                self.weight[14] = 10.260489463806	
                self.weight[15] = 10.287888526917	
                self.weight[16] = 10.289801597595	
                self.weight[17] = 10.405355453491	
                self.weight[18] = 10.138095855713
            else:
                self.weight[0] = 2.9824
                self.weight[1] = 8.3159
                self.weight[2] = 5.0156
                self.weight[3] = 10.0102
                self.weight[4] = 9.8439
                self.weight[5] = 10.0466
                self.weight[6] = 10.4210
                self.weight[7] = 10.1949
                self.weight[8] = 5.5653
                self.weight[9] = 9.8416
                self.weight[10] = 8.2635
                self.weight[11] = 10.2304
                self.weight[12] = 10.4549
                self.weight[13] = 7.4082
                self.weight[14] = 10.3961
                self.weight[15] = 10.3649
                self.weight[16] = 10.4143
                self.weight[17] = 10.4677
                self.weight[18] = 10.3924
            self.weight[19] = 0
        elif dataset == "camvid":
            self.weight = torch.ones(12)
            if only_encode:
                self.weight[0] = 4.2105
                self.weight[1] = 3.4635
                self.weight[2] = 9.5835
                self.weight[3] = 2.8925
                self.weight[4] = 7.2272
                self.weight[5] = 5.5522
                self.weight[6] = 9.4418
                self.weight[7] = 9.4783
                self.weight[8] = 6.6290
                self.weight[9] = 9.8325
                self.weight[10] = 10.2083
            else:
                self.weight[0] = 4.2000
                self.weight[1] = 3.4628
                self.weight[2] = 9.5914
                self.weight[3] = 2.8961
                self.weight[4] = 7.2291
                self.weight[5] = 5.5525
                self.weight[6] = 9.4410
                self.weight[7] = 9.4781
                self.weight[8] = 6.6297
                self.weight[9] = 9.8337
                self.weight[10] = 10.2085
            self.weight[11] = 0
        else:
            print("Unsupported dataset!")

            