import sys
sys.path.append('..')
import os
import torch
import torch.nn.functional as F
import time
from torch.nn import CrossEntropyLoss, L1Loss
from utils.train_utils import DiceLoss, plot_loss
from torch.optim.lr_scheduler import PolynomialLR
from medpy.metric.binary import dc
from torchinfo import summary
import matplotlib
matplotlib.use('Agg')

def Trainer(train_dataset, val_dataset, model, config):
    torch.manual_seed(420)
    torch.backends.cudnn.benchmark = True

    device = config['DEVICE']
    model = model.to(device)
    summary(model, (config['BATCH_SIZE'], config['IMAGE_DIM'], config['RESOLUTION'], config['RESOLUTION']))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Number of Trainable Params: ', pytorch_total_params)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True,num_workers=config['WORKERS'])
    print('Number of Train Examples: ', train_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = config['VAL_BATCH_SIZE'], num_workers=config['WORKERS'])
    print('Number of Val Examples: ', val_dataset.__len__())

    optimizer = torch.optim.SGD(params = model.parameters(), lr = config['LR'], momentum = 0.99, nesterov=True)

    train_loss_avg = []
    train_seg_avg = []
    train_bbox_avg = []
    train_dice_lumen_avg = []
    train_dice_eem_avg = []
    val_loss_avg = []
    val_seg_avg = []
    val_bbox_avg = []
    val_dice_lumen_avg = []
    val_dice_eem_avg = []

    folder = os.path.join(config['DIR']+"Results", config['NAME'])
    try:
        os.mkdir(folder)
        os.mkdir(os.path.join(folder, 'train'))
        os.mkdir(os.path.join(folder, 'val'))
    except:
        pass

    best_dice = 0
    
    print('Training ...')

    if config['LOSS'] == 'dice_ce':
        dice_loss = DiceLoss().to(device)
        ce_loss = CrossEntropyLoss().to(device)
    else:
        print('no loss selected')

    if config['BBOX_LOSS'] == 'l1':
        bbox_l1_loss = L1Loss()
    else:
        print('no bbox loss selected')

    scheduler = PolynomialLR(optimizer, total_iters=config['EPOCHS'], power=0.9)

    for epoch in range(config['EPOCHS']):

        start = time.time()

        model.train()

        train_loss_avg.append(0)
        train_seg_avg.append(0)
        train_bbox_avg.append(0)
        train_dice_lumen_avg.append(0)
        train_dice_eem_avg.append(0)
        num_batches = 0

        for batch in train_loader:

            image, mask, bbox, id = batch['image'].to(device), batch['mask'].to(device), batch['bbox'].to(device), batch['id']

            if config['RASTER_SIZE'] != [config['MASK_SIZE'], config['MASK_SIZE'], config['MASK_SIZE'],config['MASK_SIZE']]:
                masks = [F.interpolate(mask.unsqueeze(1), (config['RASTER_SIZE'][i], config['RASTER_SIZE'][i]), mode='bilinear', align_corners=True)[:,0,:,:] for i in range(4)]
            else:
                masks = [mask, mask, mask, mask]

            out = model(image)
            seg_list, coord_list, bbox_pred = out

            optimizer.zero_grad()

            bbox_loss = bbox_l1_loss(bbox_pred, bbox) * config['BBOX_LOSS_WEIGHT']
            seg_loss = []
            for i, (seg_pred, mask) in enumerate(zip(seg_list, masks)):
                wi = config['LOSS_WEIGHT'][i]
                mask = mask.long()
                seg_loss.append((dice_loss(seg_pred, mask) + ce_loss(seg_pred, mask)) * wi)

            seg_total_loss = torch.stack(seg_loss).sum()
            total_loss = bbox_loss + seg_total_loss

            # final output
            segs = [torch.argmax(seg_list[i], dim=1).cpu().numpy() for i in range(len(seg_list))]
            mask = masks[-1].cpu().numpy()

            dice_lumen = dc(segs[-1] == 2, mask == 2)
            dice_eem = dc(segs[-1] != 0, mask != 0)

            train_loss_avg[-1] += total_loss.item()
            train_seg_avg[-1] += seg_total_loss.item()
            train_bbox_avg[-1] += bbox_loss.item()
            train_dice_lumen_avg[-1] += dice_lumen
            train_dice_eem_avg[-1] += dice_eem

            total_loss.backward()

            optimizer.step()

            num_batches += 1
            if num_batches == 250:
                break

        train_loss_avg[-1] /= num_batches
        train_seg_avg[-1] /= num_batches
        train_bbox_avg[-1] /= num_batches
        train_dice_lumen_avg[-1] /= num_batches
        train_dice_eem_avg[-1] /= num_batches

        print('Epoch [%d / %d] Train Loss: %0.4f (Seg %0.4f, BBox %0.4f) DC Lumen %0.4f EEM %0.4f ' % (
            epoch + 1, config['EPOCHS'], train_loss_avg[-1], train_seg_avg[-1], train_bbox_avg[-1], train_dice_lumen_avg[-1], train_dice_eem_avg[-1]))

        num_batches = 0

        model.eval()
        val_dice_lumen_avg.append(0)
        val_dice_eem_avg.append(0)
        val_loss_avg.append(0)
        val_seg_avg.append(0)
        val_bbox_avg.append(0)

        with torch.no_grad():
            for batch in val_loader:

                image, mask, bbox, id = batch['image'].to(device), batch['mask'].to(device), batch['bbox'].to(device), batch['id']

                if config['RASTER_SIZE'] != [config['MASK_SIZE'], config['MASK_SIZE'], config['MASK_SIZE'],
                                               config['MASK_SIZE']]:
                    masks = [F.interpolate(mask.unsqueeze(1), (config['RASTER_SIZE'][i], config['RASTER_SIZE'][i]),
                                           mode='bilinear', align_corners=True)[:, 0, :, :] for i in range(4)]
                else:
                    masks = [mask, mask, mask, mask]


                out = model(image)
                seg_list, coord_list, bbox_pred = out

                bbox_loss = bbox_l1_loss(bbox_pred, bbox) * config['BBOX_LOSS_WEIGHT']
                seg_loss = []
                for i, (seg_pred, mask) in enumerate(zip(seg_list, masks)):
                    wi = config['LOSS_WEIGHT'][i]
                    mask = mask.long()
                    seg_loss.append((dice_loss(seg_pred, mask) + ce_loss(seg_pred, mask)) * wi)
                seg_total_loss = torch.stack(seg_loss).sum()
                total_loss = bbox_loss + seg_total_loss

                segs = [torch.argmax(seg_list[i], dim=1).cpu().numpy() for i in range(len(seg_list))]
                mask = masks[-1].cpu().numpy()

                dice_lumen = dc(segs[-1] == 2, mask == 2)
                dice_eem = dc(segs[-1] != 0, mask != 0)

                val_loss_avg[-1] += total_loss.item()
                val_seg_avg[-1] += seg_total_loss.item()
                val_bbox_avg[-1] += bbox_loss.item()
                val_dice_lumen_avg[-1] += dice_lumen
                val_dice_eem_avg[-1] += dice_eem
                num_batches += 1
                loss_rec = 0

        val_loss_avg[-1] /= num_batches
        val_seg_avg[-1] /= num_batches
        val_bbox_avg[-1] /= num_batches
        val_dice_lumen_avg[-1] /= num_batches
        val_dice_eem_avg[-1] /= num_batches

        print('Epoch [%d / %d] Val Loss: %0.4f (Seg %0.4f, BBox %0.4f) DC Lumen %0.4f EEM %0.4f' % (
            epoch + 1, config['EPOCHS'], val_loss_avg[-1], val_seg_avg[-1], val_bbox_avg[-1], val_dice_lumen_avg[-1], val_dice_eem_avg[-1]))

        end = time.time()
        epoch_time = end - start
        print('Epoch time: {:.2f}s'.format(epoch_time))

        if config['SAVE_EVERY_EPOCH']:
            print('Model Saved')
            out = str(epoch)+".pt"
            torch.save(model.state_dict(), os.path.join(folder, out))

        else:
            if (val_dice_lumen_avg[-1]+val_dice_eem_avg[-1]) > best_dice:
                best_dice = val_dice_lumen_avg[-1] + val_dice_eem_avg[-1]
                print('Model Saved Dice')
                torch.save(model.state_dict(), os.path.join(folder, "bestDice.pt"))

        scheduler.step()

        plot_loss(train_loss_avg, val_loss_avg,
                  config['EPOCHS'], epoch,
                  folder + '/train_val_loss.png', y=(0, 0.1, 0.005))

        plot_loss(train_dice_lumen_avg, val_dice_eem_avg,
                  config['EPOCHS'], epoch,
                  folder + '/train_val_dice_lumen.png', y=(0.9, 1, 0.01))

        plot_loss(train_dice_eem_avg, val_dice_eem_avg,
                  config['EPOCHS'], epoch,
                  folder + '/train_val_dice_eem.png', y=(0.9, 1, 0.01))

