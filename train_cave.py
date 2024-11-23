

from scipy.io import loadmat
from thop import profile, clever_format

from LRTN import Cross_Guide_Fusion
from calculate_metrics import Loss_SAM, Loss_RMSE, Loss_PSNR

from train_dataloader import *
from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
from utils import create_F, fspecial
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def mkdir(path):

    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("训练文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))


if __name__ == '__main__':
    # 路径参数
    root=os.getcwd()+"/CAVE"
    model_name='LRTN'
    mkdir(os.path.join(root,model_name))
    current_list=os.listdir(os.path.join(root,model_name))
    for i in current_list:
        if len(i)>1:
            current_list.remove(i)

    current_list.append('0')
    int_list = [int(x) for x in current_list]
    train_value = max(int_list)+1
    model_name=os.path.join(model_name,str(train_value))

    data_name='cave'
    path1 = r'D:\data\CAVE\train/'
    path2 = r'D:\data\CAVE\test/'



    # 训练参数
    loss_func = nn.L1Loss(reduction='mean').cuda()
    R = create_F()
    PSF = fspecial('gaussian', 8, 3)
    downsample_factor = 8
    training_size = 64   # 训练和测试裁剪块的大小
    stride = 32   # 滑动窗口方式裁剪，stride
    stride1 = 32   # 重建用的stride
    LR = 4e-4
    EPOCH =2000
    weight_decay=1e-8
    BATCH_SIZE = 16
    num = 20
    psnr_optimal = 44
    rmse_optimal = 1.5
    val_epoch=100                     # 前100epoc不测试
    val_interval = 50           # 每隔val_interval epoch测试一次
    checkpoint_interval = 100
    # maxiteration = math.ceil(((512 - training_size) // stride + 1) ** 2 * num / BATCH_SIZE) * EPOCH




    a,b=64,64
    hsi = torch.randn(2, 31,a//8,b//8).cuda()
    msi = torch.randn(2, 3, a,b).cuda()
    model2 = Cross_Guide_Fusion(31, a,b,1).cuda()
    flops, params,DIC= profile(model2, inputs=(hsi,msi),ret_layer_info=True)
    flops, params= clever_format([flops, params], "%.3f")
    print(flops, params)


    # 创建方法名字的文件
    path=os.path.join(root,model_name)
    mkdir(path)  # 创建文件夹
    # 创建训练记录文件
    pkl_name=data_name+'_pkl'
    pkl_path=os.path.join(path,pkl_name)      # 模型保存路径
    os.makedirs(pkl_path)      # 创建文件夹
    # 创建excel
    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss','val_loss','val_rmse', 'val_psnr', 'val_sam'])  # 列名
    excel_name=data_name+'_record.csv'
    excel_path=os.path.join(path,excel_name)
    df.to_csv(excel_path, index=False)


    train_data = CAVEHSIDATAprocess(path1, R, training_size, stride, downsample_factor, PSF, num)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


    maxiteration = math.ceil(len(train_data) / BATCH_SIZE) * EPOCH
    print("maxiteration：", maxiteration)

    cnn = Cross_Guide_Fusion(31, training_size, training_size,1).cuda()


    # 模型初始化
    for m in cnn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR,betas=(0.9, 0.999),weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxiteration, eta_min=0, last_epoch=-1)


    start_epoch = 0




    for epoch in range(start_epoch+1, EPOCH+1):
        cnn.train()
        loss_all = []
        loop = tqdm(train_loader, total=len(train_loader))
        for a1, a2, a3 in loop:
            lr = optimizer.param_groups[0]['lr']
            output = cnn(a2.cuda(),a3.cuda())
            loss = loss_func(output, a1.cuda())
            loss_temp = loss
            loss_all.append(np.array(loss_temp.detach().cpu().numpy()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_postfix({'loss': '{0:1.8f}'.format(np.mean(loss_all)), "lr": '{0:1.8f}'.format(lr)})
            scheduler.step()





        if ((epoch % val_interval == 0) and (epoch >= val_epoch)) or epoch==1 or epoch==EPOCH:
            cnn.eval()
            val_loss=AverageMeter()
            SAM = Loss_SAM()
            RMSE = Loss_RMSE()
            PSNR = Loss_PSNR()
            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()

            imglist = os.listdir(path2)
            with torch.no_grad():
                for i in range(0, len(imglist)):
                    img = h5.loadmat(path2 + imglist[i])
                    img1 = img["b"]
                    img1 = img1 / img1.max()
                    HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))

                    w, h = int(HRHSI.shape[1] / downsample_factor), int(HRHSI.shape[2] / downsample_factor)
                    HRHSI = HRHSI[:, :w * downsample_factor, :h * downsample_factor]

                    MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
                    HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
                    MSI_1 = torch.unsqueeze(MSI, 0)
                    HSI_LR1 = torch.unsqueeze(torch.Tensor(HSI_LR), 0)  # 加维度 (b,c,h,w)
                    # 计算val_loss用的，防止出错单独拿出来
                    to_fet_loss_hr_hsi=torch.unsqueeze(torch.Tensor(HRHSI), 0)

                    prediction,val_loss = reconstruction(cnn, HSI_LR1.cuda(), MSI_1.cuda(), to_fet_loss_hr_hsi,downsample_factor, training_size, stride1,val_loss)
                    # print(Fuse.shape)
                    sam.update(SAM(np.transpose(HRHSI.cpu().detach().numpy(),(1, 2, 0)),np.transpose(prediction.squeeze().cpu().detach().numpy(),(1, 2, 0))))
                    rmse.update(RMSE(HRHSI.cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))
                    psnr.update(PSNR(HRHSI.cpu().permute(1,2,0),prediction.squeeze().cpu().permute(1,2,0)))

                if  epoch == 1:
                    torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_PSNR_best.pkl')

                if torch.abs(psnr_optimal-psnr.avg)<0.15:
                    torch.save(cnn.state_dict(), pkl_path + '/' + str(epoch) + 'EPOCH' + '_PSNR_best.pkl')
                if psnr.avg > psnr_optimal:
                    psnr_optimal = psnr.avg

                if torch.abs(rmse.avg-rmse_optimal)<0.15:
                    torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_RMSE_best.pkl')
                if rmse.avg < rmse_optimal:
                    rmse_optimal = rmse.avg



                print("val  PSNR:",psnr.avg.cpu().detach().numpy(), "  RMSE:", rmse.avg.cpu().detach().numpy(), "  SAM:", sam.avg,"val loss:", val_loss.avg.cpu().detach().numpy())
                val_list = [epoch, lr,np.mean(loss_all),val_loss.avg.cpu().detach().numpy(),rmse.avg.cpu().detach().numpy(), psnr.avg.cpu().detach().numpy(), sam.avg]

                val_data = pd.DataFrame([val_list])
                val_data.to_csv(excel_path,mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
                time.sleep(0.1)

