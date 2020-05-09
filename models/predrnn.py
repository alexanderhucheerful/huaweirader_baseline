#author:alexhu
#time:2020.5.6

import torch
import torch.nn as nn

gpu_batch = 8
class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride,batch =8):
        super(SpatioTemporalLSTMCell, self).__init__()
        batch = batch
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 7, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)


    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


    def init_hidden(self, batch_size, image_size):

        height, width = image_size,image_size

        return [torch.zeros(batch_size, self.num_hidden, height, width, device=self.conv_last.weight.device),

                torch.zeros(batch_size, self.num_hidden, height, width, device=self.conv_last.weight.device)]



class  CausalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride):
        super(CausalLSTMCell, self).__init__()
        
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.GroupNorm(7, num_hidden*7))
        
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.GroupNorm(4, num_hidden*4))
        
        self.conv_c = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.GroupNorm(3, num_hidden*3))

        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.GroupNorm(3, num_hidden*3))

        self.conv_cnew = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden*4 , kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.GroupNorm(4, num_hidden*4))

        self.conv_mnew = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden , kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.GroupNorm(1, num_hidden*1))

        self.conv_cell = nn.Sequential(
            nn.Conv2d(num_hidden*2, num_hidden , kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.GroupNorm(1, num_hidden*1))
        
        self.forget_bias = 1.0

    def forward(self, x_t, h_t, c_t, m_t):
        
        h_cc = self.conv_h(h_t)
        x_cc = self.conv_x(x_t)
        c_cc = self.conv_c(c_t)
        m_cc = self.conv_m(m_t)
        
        i_h, g_h, f_h, o_h = torch.split(h_cc, self.num_hidden, dim=1)
        i_c, g_c, f_c = torch.split(c_cc, self.num_hidden, dim=1)
        i_m, f_m, m_m = torch.split(m_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = torch.split(x_cc,self.num_hidden, dim=1)
        
        
        i = torch.sigmoid(i_x + i_h + i_c)
        f = torch.sigmoid(f_x + f_h + f_c + self.forget_bias)
        g = torch.tanh(g_x + g_h + g_c)
        
        c_new = f * c_t + i * g
        #print(c_new.shape)
        
        c2m = self.conv_cnew(c_new)
        i_c, g_c, f_c, o_c = torch.split(c2m, self.num_hidden,dim=1)
        
        
        ii = torch.sigmoid(i_c + i_x_ + i_m)
        ff = torch.sigmoid(f_c + f_x_ + f_m + self.forget_bias)
        gg = torch.tanh(g_c + g_x_)
        
        m_new = ff * torch.tanh(m_m) + ii * gg
        
        o_m = self.conv_mnew(m_new)
        
        o = torch.tanh(o_x + o_h + o_c + o_m)
        
        
        cell = torch.cat((c_new, m_new), 1)
        cell = self.conv_cell(cell)
        h_new = o * torch.tanh(cell)
        
        return h_new, c_new, m_new

    def init_hidden(self, batch_size, image_size):

        height, width = image_size,image_size

        return [torch.zeros(batch_size, self.num_hidden, height, width, device=self.conv_h[0].weight.device),

                torch.zeros(batch_size, self.num_hidden, height, width, device=self.conv_h[0].weight.device)]


class GhuCell(nn.Module):
    def __init__(self,in_channel,filter_size,stride):
        super(GhuCell,self).__init__()
        
        self.in_channel = in_channel
        self.padding = filter_size // 2
        self.layer1 = nn.Conv2d(in_channel,in_channel*2,filter_size,stride,self.padding)
        self.layer2 = nn.Conv2d(in_channel,in_channel*2,filter_size,stride,self.padding)
    
    def forward(self,h,z):
        if z is None:
            z = torch.zeros(h.size(0),h.size(1),h.size(2),h.size(3),device=h.device)
        z_concat = self.layer1(z)
        h_concat = self.layer2(h)
        p,u = torch.split(torch.add(z_concat,h_concat),self.in_channel,dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u*p + (1-u)*z
        return z_new,z_new
    
"""
model = CausalLSTMCell(1,32,64,3,1)
model = model.cuda()
x = torch.randn(2,1,64,64).cuda()
h = torch.randn(2,32,64,64).cuda()
c = torch.randn(2,32,64,64).cuda()
m = torch.randn(2,32,64,64).cuda()

hx,cx,mx = model(x,h,c,m)
"""
      
"""
convlayer1 model =  nn.Conv2d (5,4,1)  256*256->64*64 1-16
convlayer2 model =  nn.Conv2d (3,2,1)  64*64 -> 32*32 16-64
convlayer3 model =  nn.Conv2d (3,2,1)  32*32 ->16*16   64-128

deconvlayer
layer1 = nn.ConvTranspose2d(128,64,4,2,1)
layer2 = nn.ConvTranspose2d(64,16,4,2,1)
#可以采用级联的方式融合
layer3 = nn.ConvTranspose2d(16,1,6,4,1)


"""

class predrnn_encoder(nn.Module):
    """
    写成伪静态图的形式，不方便改(其实也很好改)但容易理解
    3layers，
    """
    def __init__(self,model = 'encoder',numlayers = 3,num_hidden = None):
        super(predrnn_encoder,self).__init__()
        self.num_hidden = (16,32,64)
        self.num_layers = numlayers
        #m和h分开下采样
        cell_list = [nn.Conv2d(1,16,5,4,1,bias = False),
                     CausalLSTMCell(16,16,64,3,1), #1
                     GhuCell(64,3,1),
                     nn.Conv2d(16,64,3,2,1,bias = False),#3
                     nn.Conv2d(16,64,3,2,1,bias = False),
                     SpatioTemporalLSTMCell(64,64,32,3,1), 
                     nn.Conv2d(64,128,3,2,1,bias = False),#6
                     nn.Conv2d(64,128,3,2,1,bias = False),
                     SpatioTemporalLSTMCell(128,128,16,3,1)] 

        self.cell_list = nn.ModuleList(cell_list)

        self.convert_mstate = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1),
                                            nn.ConvTranspose2d(64,16,4,2,1))

    def forward(self,input_tensor_org,hidden_state = None,m_state = None):

        batchsize,seq,input_channel,height,width = input_tensor_org.size()
        #seq = input_tensor.shape[1]

        #m显然只在encoder初始化一次即可
        #print("fuck")
        if m_state == None:
            m_state = self.cell_list[1].init_hidden(batchsize,64)[0]
        else:
            m_state = m_state

        if hidden_state == None:
            hidden_state = [self.cell_list[1].init_hidden(batchsize,64),
                            self.cell_list[5].init_hidden(batchsize,32),
                            self.cell_list[8].init_hidden(batchsize,16)
                            ]
        
        #next_hidden=[]
        #next_m = []
        outputhidden = []

        assert seq ==10
        z =  None
        #print("that ok")
        for t in range(seq):
            #print('t time',t)
            input_tensor = input_tensor_org[:,t,...]
            input_tensor = torch.reshape(input_tensor,(-1,input_channel,height,width))
            input_tensor = self.cell_list[0](input_tensor)
            input_tensor = torch.reshape(input_tensor,(batchsize,input_tensor.size(1),input_tensor.size(2),input_tensor.size(3)))
            if t == 0 :
                hidden_state[0][0],hidden_state[0][1],m_state = self.cell_list[1](input_tensor,hidden_state[0][0],hidden_state[0][1],m_state)
            else:
                #convert the m to lowlayer size
                m_state = self.convert_mstate(m_state)
                hidden_state[0][0],hidden_state[0][1],m_state = self.cell_list[1](input_tensor,hidden_state[0][0],hidden_state[0][1],m_state)
            
            #inputh,z = self.cell_list[2](hidden_state[0][0],z)
            for layer in range(1,self.num_layers):                    
                input_hidden = hidden_state[layer-1][0]
                input_hidden = self.cell_list[layer*3](input_hidden)

                m_state = self.cell_list[layer*3+1](m_state)
                #if layer == self.numlayers:
                #print('layer',layer)
                if layer == 1:
                    inputh,z = self.cell_list[2](input_hidden,z)
                    input_hidden = inputh
                    #print(z)

                hidden_state[layer][0],hidden_state[layer][0],m_state = self.cell_list[layer*3+2](input_hidden,hidden_state[layer][0],hidden_state[layer][1],m_state)
        
        next_m = m_state
        return hidden_state,next_m,z




class predrnn_decoder(nn.Module):
    """
    写成伪静态图的形式，不方便改(其实也很好改)但容易理解
    decoder顺序是encoder顺序倒着写
    3layers，
    """
    def __init__(self,model = 'decoder',numlayers = 3,num_hidden = None):
        super(predrnn_decoder,self).__init__()
        self.num_hidden = (128,64,128)
        self.num_layers = numlayers
        #m和h分开上采样
        cnn_cell_list = [nn.ConvTranspose2d(128,64,4,2,1),
                         nn.ConvTranspose2d(128,64,4,2,1),
                         nn.ConvTranspose2d(64,16,4,2,1),
                         nn.ConvTranspose2d(64,16,4,2,1)
                        ]

        rnn_cell_list = [SpatioTemporalLSTMCell(128,128,16,3,1),
                        SpatioTemporalLSTMCell(64,64,32,3,1),
                        CausalLSTMCell(16,16,64,3,1)
                        ]    

        ghu_cell_list = [GhuCell(64,3,1)] 



    

        self.cnn_cell_list = nn.ModuleList(cnn_cell_list)
        self.rnn_cell_list = nn.ModuleList(rnn_cell_list)
        self.ghu_cell_list = nn.ModuleList(ghu_cell_list)
        """
        self.convert_mstate = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1),
                                            nn.ConvTranspose2d(64,16,4,2,1))
        """

        self.convert_mstate = nn.Sequential(nn.Conv2d(16,64,3,2,1),
                                            nn.Conv2d(64,128,3,2,1))

        self.lastoutput = nn.Sequential(nn.ConvTranspose2d(16,8,6,4,1),
                                        nn.Conv2d(8,4,3,1,1),
                                        nn.Conv2d(4,1,1,1,0))
        

    def forward(self,input_tensor,hidden_state,m_state,z):

        #batchsize,seq,input_channel,height,width = input_tensor.size()
        #seq = input_tensor.shape[1]
        #显然在decoder初始化的时候input_tensor是none，用每一个timestep的输出作为下一个的输入
        hidden_state = hidden_state[::-1]
        #print(hidden_state[0][0].shape,hidden_state[1][0].shape,hidden_state[2][0].shape)
        input_tensor = input_tensor
        if input_tensor == None:
            input_tensor = self.rnn_cell_list[0].init_hidden(hidden_state[0][0].shape[0],16)[0]
        z = z
        #print("fix the bug")
        """
        #m显然只在encoder初始化一次即可
        if m_state == None:
            m_state = self.cell_list[1].init_hidden(batchsize,64)[0]
        else:
            m_state = m_state

        if hidden_state == None:
            hidden_state = [self.cell_list[1].init_hidden(batchsize,64),
            hidden_state =  self.cell_list[5].init_hidden(batchsize,32),
            hidden_state =  self.cell_list[8].init_hidden(batchsize,16)
                            ]
        """
        #next_hidden=[]
        #next_m = []
        hidden_state = hidden_state
        m_state = m_state
        outputframes = []

        seq =4
        for t in range(seq):
            #print('time',t)
            input_tensor = input_tensor

            if t == 0 :
                hidden_state[0][0],hidden_state[0][1],m_state = self.rnn_cell_list[0](input_tensor,hidden_state[0][0],hidden_state[0][1],m_state)
            else:
                #convert the m to lowlayer size
                m_state = self.convert_mstate(m_state)
                hidden_state[0][0],hidden_state[0][1],m_state = self.rnn_cell_list[0](input_tensor,hidden_state[0][0],hidden_state[0][1],m_state)

            
            for layer in range(1,self.num_layers):
                #print("layer",layer)
                input_hidden = hidden_state[layer-1][0]
                if layer ==self.num_layers-1:
                    input_hidden,z = self.ghu_cell_list[0](input_hidden,z)
                input_hidden = self.cnn_cell_list[(layer-1)*2](input_hidden)
                m_state = self.cnn_cell_list[((layer-1)*2+1)](m_state)
                """
                if layer ==self.num_layers-1:
                    input_hidden,z = self.ghu_cell_list[0](input_hidden,z)
                """
                    
                hidden_state[layer][0],hidden_state[layer][0],m_state = self.rnn_cell_list[layer](input_hidden,hidden_state[layer][0],hidden_state[layer][1],m_state)
            outframe = self.lastoutput(hidden_state[-1][0])
            outputframes.append(outframe)
        
        out = torch.stack(outputframes,dim=1)
        print(out.shape)
        
        return out



class predrnned(nn.Module):
    def __init__(self,model = 'predrnn++'):
        super(predrnned,self).__init__()
        
        self.encoder = predrnn_encoder()
        self.decoder = predrnn_decoder()

    def forward(self,input_tensor):
        input_tensor = input_tensor
        #print(self.encoder)
        h,m,z= self.encoder(input_tensor,None,None)
        #print('encoder passing unity testing')
        input_tensor = None
        out = self.decoder(input_tensor,h,m,z)

        return (out,)


if __name__ == '__main__':
    inputx = torch.rand(2,10,1,256,256).cuda(2)

    model = predrnned().cuda(2)
    target = torch.rand(2,4,1,256,256).cuda(2)
    #out =  model(inputx)
    #print(out.shape)
    lossfunc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1000):
        out =  model(inputx)
        loss = lossfunc(out,target)
        optimizer.zero_grad()  
        loss.backward()   
        optimizer.step()   
        print(epoch, 'loss:', loss.item())

