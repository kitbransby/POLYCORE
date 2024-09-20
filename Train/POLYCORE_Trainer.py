import sys
sys.path.append('..')
import os
import torch
import time
import skimage
from torch.nn import MSELoss
from utils.train_utils import plot_loss
from torch.optim.lr_scheduler import PolynomialLR
from medpy.metric.binary import dc
from torchinfo import summary
import matplotlib
matplotlib.use('Agg')

def get_grid():
    xs = torch.arange(0, 480, 1)
    ys = torch.arange(0, 480, 1)
    xx, yy = torch.meshgrid(xs, ys)
    grid = torch.stack([xx, yy], dim=-1).to(torch.float32).reshape(-1,2)
    return grid

def get_dense_maps(mask, grid):
    bs = mask.shape[0]

    vec_batched, dist_batched = torch.zeros((bs,480,480,2), device='cuda', dtype=torch.float32), torch.zeros((bs, 480,480), device='cuda', dtype=torch.float32)

    for i in range(bs):
        edges = skimage.feature.canny(  # we can also implement this in pytorch and on gpu
            image=mask[i],
            sigma=1,
            low_threshold=0.5,
            high_threshold=1.5,
        )

        edges = torch.tensor(edges, dtype=torch.float32, device='cuda')
        contour = torch.where(edges == 1)
        contour = torch.stack([contour[0], contour[1]], dim=-1).to(torch.float32)

        dist = torch.cdist(grid, contour) # for every pixel, calc distance to all contour points

        closet_pos = contour[torch.argmin(dist, dim=-1)] # for every pixel, find the closest point on contour

        vec = closet_pos - grid # for every pixel, calc direction to contour
        vec = vec / (torch.unsqueeze(torch.linalg.norm(vec, dim=-1), -1)) # normalise the direction to have mag = 1
        vec[torch.where(vec.isnan())] = 0 # where nan (i.e on boundary) = 0
        vec = vec.reshape(480,480,2)

        dist = torch.min(dist, dim=-1)[0].reshape(480,480) / 310  # the max dist in this dataset is 310px, so normalise to [0,1] by dividing by 310.

        vec = torch.stack([vec[:,:,1], vec[:,:,0]], dim=-1)

        vec_batched[i,:,:,:] = vec
        dist_batched[i,:,:] = dist

    return vec_batched, dist_batched

def Trainer(train_dataset, val_dataset, model, config):
    torch.manual_seed(420)
    torch.backends.cudnn.benchmark = True
    device = config['DEVICE']

    weight_dir = os.path.join(config['DIR'] + "Results", config['WEIGHTS'], 'bestDice.pt')
    model.load_state_dict(torch.load(weight_dir), strict=False)

    grid = get_grid()
    grid = grid.to(device)

    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.REFINEMENT.parameters():
        param.requires_grad = True

    summary(model, (config['BATCH_SIZE'], config['IMAGE_DIM'], config['RESOLUTION'], config['RESOLUTION']))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Number of Trainable Params: ', pytorch_total_params)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True,
                                               num_workers=8)
    print('Number of Train Examples: ', train_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['VAL_BATCH_SIZE'], num_workers=8)
    print('Number of Val Examples: ', val_dataset.__len__())

    optimizer = torch.optim.SGD(params=model.parameters(), lr=config['LR'], momentum=0.99, nesterov=True)

    train_loss_avg = []
    train_vec_loss_avg = []
    train_dist_loss_avg = []
    train_dice_lumen_avg = []
    train_dice_eem_avg = []
    val_loss_avg = []
    val_vec_loss_avg = []
    val_dist_loss_avg = []
    val_dice_lumen_avg = []
    val_dice_eem_avg = []

    folder = os.path.join(config['DIR'] + "Results", config['NAME'])
    try:
        os.mkdir(folder)
        os.mkdir(os.path.join(folder, 'train'))
        os.mkdir(os.path.join(folder, 'val'))
    except:
        pass

    best_dice = 0
    mse_loss = MSELoss()
    scheduler = PolynomialLR(optimizer, total_iters=config['EPOCHS'], power=0.9)

    print('Training ...')
    for epoch in range(config['EPOCHS']):

        start = time.time()

        model.train()

        train_loss_avg.append(0)
        train_vec_loss_avg.append(0)
        train_dist_loss_avg.append(0)
        train_dice_lumen_avg.append(0)
        train_dice_eem_avg.append(0)
        num_batches = 0

        for batch in train_loader:

            image, mask, bbox, id = batch['image'].to(device), batch['mask'].to(device), batch['bbox'].to(device), batch['id']

            lumen_vec, lumen_dist = get_dense_maps(mask.cpu().numpy() == 2, grid)
            eem_vec, eem_dist = get_dense_maps(mask.cpu().numpy() != 0, grid)
            vec = torch.stack([eem_vec, lumen_vec], dim=1)
            dist = torch.stack([eem_dist, lumen_dist], dim=1)

            out = model(image)
            refined_pred, refined_pts, vec_pred, dist_pred, coord_list, bbox_pred = out

            optimizer.zero_grad()

            vec_loss = mse_loss(vec_pred, vec)
            dist_loss = mse_loss(dist_pred, dist)  #* 10

            total_loss = vec_loss + dist_loss

            seg = torch.argmax(refined_pred.detach(), dim=1).cpu().numpy()
            mask = mask.cpu().numpy()

            dice_lumen = dc(seg == 2, mask == 2)
            dice_eem = dc(seg != 0, mask != 0)

            train_loss_avg[-1] += total_loss.item()
            train_dist_loss_avg[-1] += dist_loss.item()
            train_vec_loss_avg[-1] += vec_loss.item()
            train_dice_lumen_avg[-1] += dice_lumen
            train_dice_eem_avg[-1] += dice_eem

            total_loss.backward()

            optimizer.step()

            num_batches += 1
            if num_batches == 250:
                break

        train_loss_avg[-1] /= num_batches
        train_vec_loss_avg[-1] /= num_batches
        train_dist_loss_avg[-1] /= num_batches
        train_dice_lumen_avg[-1] /= num_batches
        train_dice_eem_avg[-1] /= num_batches

        print('Epoch [%d / %d] Train Loss: %0.4f (Dist: %0.4f, Vec %0.4f) DC Lumen %0.4f EEM %0.4f ' % (
            epoch + 1, config['EPOCHS'], train_loss_avg[-1], train_dist_loss_avg[-1], train_vec_loss_avg[-1],
            train_dice_lumen_avg[-1], train_dice_eem_avg[-1]))

        num_batches = 0

        model.eval()
        val_loss_avg.append(0)
        val_dist_loss_avg.append(0)
        val_vec_loss_avg.append(0)
        val_dice_lumen_avg.append(0)
        val_dice_eem_avg.append(0)

        with torch.no_grad():
            for batch in val_loader:
                image, mask, bbox, id = batch['image'].to(device), batch['mask'].to(device), batch['bbox'].to(device), batch['id']

                lumen_vec, lumen_dist = get_dense_maps(mask.cpu().numpy() == 2, grid)
                eem_vec, eem_dist = get_dense_maps(mask.cpu().numpy() != 0, grid)
                vec = torch.stack([eem_vec, lumen_vec], dim=1)
                dist = torch.stack([eem_dist, lumen_dist], dim=1)
                out = model(image)

                refined_pred, refined_pts, vec_pred, dist_pred, coord_list, bbox_pred = out

                optimizer.zero_grad()

                vec_loss = mse_loss(vec_pred, vec)
                dist_loss = mse_loss(dist_pred, dist)

                total_loss = vec_loss + dist_loss

                seg = torch.argmax(refined_pred.detach(), dim=1).cpu().numpy()
                mask = mask.cpu().numpy()

                dice_lumen = dc(seg == 2, mask == 2)
                dice_eem = dc(seg != 0, mask != 0)

                val_loss_avg[-1] += total_loss.item()
                val_vec_loss_avg[-1] += vec_loss.item()
                val_dist_loss_avg[-1] += dist_loss.item()
                val_dice_lumen_avg[-1] += dice_lumen
                val_dice_eem_avg[-1] += dice_eem
                num_batches += 1
                loss_rec = 0

        val_loss_avg[-1] /= num_batches
        val_dist_loss_avg[-1] /= num_batches
        val_vec_loss_avg[-1] /= num_batches
        val_dice_lumen_avg[-1] /= num_batches
        val_dice_eem_avg[-1] /= num_batches

        print('Epoch [%d / %d] Val Loss: %0.4f (Dist: %0.4f, Vec %0.4f) DC Lumen %0.4f EEM %0.4f' % (
            epoch + 1, config['EPOCHS'], val_loss_avg[-1], val_dist_loss_avg[-1], val_vec_loss_avg[-1],
            val_dice_lumen_avg[-1], val_dice_eem_avg[-1]))

        end = time.time()
        epoch_time = end - start
        print('Epoch time: {:.2f}s'.format(epoch_time))

        if (val_dice_lumen_avg[-1]+val_dice_eem_avg[-1]) > best_dice:
            best_dice = val_dice_lumen_avg[-1]+val_dice_eem_avg[-1]
            print('Model Saved Dice')
            torch.save(model.state_dict(), os.path.join(folder, "bestDice.pt"))

        scheduler.step()

        plot_loss(train_loss_avg, val_loss_avg,
                  config['EPOCHS'], epoch,
                  folder + '/train_val_loss.png', y=(0, 1, 0.005))

        plot_loss(train_dice_lumen_avg, val_dice_lumen_avg,
                  config['EPOCHS'], epoch,
                  folder + '/train_val_dice_lumen.png', y=(0.9, 1, 0.1))

        plot_loss(train_dice_eem_avg, val_dice_eem_avg,
                  config['EPOCHS'], epoch,
                  folder + '/train_val_dice_eem.png', y=(0.9, 1, 0.1))

