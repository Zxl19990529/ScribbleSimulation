from pycocotools.coco import COCO
import numpy as np
import os,cv2
import imgviz
from tqdm import tqdm
from utils import get_dfs_points,get_bezier_points,get_random_points,get_random_boundary_v2,decide_path_or_boundary
from skimage import morphology
from multiprocessing import Pool
import argparse
import sys # 导入sys模块
# sys.setrecursionlimit(3000) # 将默认的递归深度修改为3000

def imgs2scribble(args,imginfos,dataDir,dataType,saveDir,tqdm_id):
    
    semantic_output_dir = os.path.join(saveDir,'semantic_gt')
    if not os.path.exists(semantic_output_dir):
        os.makedirs(semantic_output_dir)
    vis_semantic_outputdir = os.path.join(saveDir, 'semantic_gt_vis')
    if not os.path.exists(vis_semantic_outputdir):
        os.makedirs(vis_semantic_outputdir)
    scribble_output_dir = os.path.join(saveDir,'scribble_gt')
    if not os.path.exists(scribble_output_dir):
        os.makedirs(scribble_output_dir)
    vis_scribble_outputdir = os.path.join(saveDir,'scribble_gt_vis')
    if not os.path.exists(vis_scribble_outputdir):
        os.makedirs(vis_scribble_outputdir)
    vis_skeleton_outputdir = os.path.join(saveDir,'skeleton_vis')
    if not os.path.exists(vis_skeleton_outputdir):
        os.makedirs(vis_skeleton_outputdir)
    vis_boundary_outputdir = os.path.join(saveDir,'boundary_vis')
    if not os.path.exists(vis_boundary_outputdir):
        os.makedirs(vis_boundary_outputdir)
    colormap = imgviz.label_colormap()
    
    for id,imginfo in tqdm(enumerate(imginfos),total=len(imginfos),ncols=70,position=tqdm_id):
        # imginfo = imginfos[209]
        filename = imginfo['file_name']
        basename = filename.split('.')[0]
        img_path = os.path.join(dataDir,dataType,filename)
        img = cv2.imread(img_path)
        
        # save the masks
        gt_path = os.path.join(semantic_output_dir,basename+'.png')
        gt_vis_path = os.path.join(vis_semantic_outputdir,basename+'.png')
        scribble_path = os.path.join(scribble_output_dir,basename+'.png')
        scribble_vis_path = os.path.join(vis_scribble_outputdir,basename+'.png')
        if os.path.isfile(gt_path):
            continue
        
        current_ann_ids = coco.getAnnIds(imgIds=imginfo['id'], catIds=catIds, iscrowd=None)
        current_anns = coco.loadAnns(current_ann_ids)
        if len(current_anns) <=0:
            continue
        height = imginfo['height']
        width = imginfo['width']
        curr_mask = np.zeros_like(img)
        curr_mask_color = np.zeros_like(img)
        scribble_canvas = np.ones_like(img) * 255
        scribble_canvas_color = np.ones_like(img) * 255
        flag = 0
        skeleton_canvas = np.zeros_like(img,dtype=np.uint8)

        # foreground
        for one_ann  in current_anns:
            if one_ann['iscrowd']:
                continue
            else:
                curr_binary_mask = coco.annToMask(one_ann)
                if np.sum(curr_binary_mask) <=5:
                    continue
                curr_binary_mask = curr_binary_mask[:,:,None].repeat(3,-1)
                curr_binary_mask = cv2.rectangle(curr_binary_mask,(1,1),(width-1,height-1),color=(0,0,0),thickness=2)
                curr_binary_mask = curr_binary_mask[:,:,0]
                curr_category = int(category_mapping[one_ann['category_id']])
                curr_mask[curr_binary_mask==1] = [curr_category,curr_category,curr_category] # h,w,3
                curr_mask_color[curr_binary_mask==1] = colormap[curr_category]
                
                # extract skeletons
                skel, distance = morphology.medial_axis(curr_binary_mask, return_distance=True)
                dist_on_skel = distance * skel # h,w
                dist_on_skel = dist_on_skel.astype(np.uint8)*255
                
                # chose the largest connected region, to remove the lone points.
                num_labels, regions, stats, centroids = cv2.connectedComponentsWithStats(dist_on_skel, connectivity=8) # regions H*W, value denotes the i-th region.
                clean_skel = 0
                largest_area = 0
                for region_id  in np.unique(regions):
                    if region_id == 0:
                        continue
                    curr_skel = (regions == region_id)
                    if np.sum(curr_skel) >=largest_area:
                        largest_area = np.sum(curr_skel)
                        clean_skel = curr_skel
                if largest_area <=2:
                    continue
                skeleton_canvas[clean_skel] = 255
                
                # max_depth_path = get_random_points(clean_skel.astype(np.int32))
                try:
                    if args.random == 'True':
                        # Choose path scribble or boundary scribble
                        # FLAG_path_or_boundary = 1 
                        FLAG_path_or_boundary = decide_path_or_boundary(curr_binary_mask) # 0 for path, 1 for boundary
                        # max_depth_path = get_dfs_points(clean_skel.astype(np.int32))
                        if FLAG_path_or_boundary: #  1 uses boundary
                            max_depth_path = get_random_boundary_v2(curr_binary_mask,jitter_pixel=20,vis_path=os.path.join(args.save_dir,'boundary_vis',basename+'.png'))
                        else: # 0 uses path
                            max_depth_path = get_random_points(clean_skel.astype(np.int32))
                    else:
                        max_depth_path = get_dfs_points(clean_skel.astype(np.int32))
                except:
                    f=open('%s.txt'%(args.dataDir),'a')
                    print(filename,one_ann,file=f)
                    f.close()
                    continue

                # if filename == filename:
                #     flag +=1
                #     print(filename,flag,id)
                # if flag == 27:
                #     print(max_depth_path)
                scribble_points = get_bezier_points(np.array(max_depth_path))
                scribble_points = scribble_points.astype(np.int32)
                scribble_canvas = cv2.polylines(scribble_canvas,scribble_points[None,:,:],False,[curr_category,curr_category,curr_category],2)
                scribble_canvas_color[scribble_canvas[:,:,0] == curr_category] = colormap[curr_category]
        
        # background mask scribble
        curr_category = 0
        background_mask = ~(curr_mask>=1) # h,w,3
        background_mask = background_mask.astype(np.uint8)
        background_mask = cv2.rectangle(background_mask,(1,1),(width-1,height-1),color=(0,0,0),thickness=2)
        skel, distance = morphology.medial_axis(background_mask[:,:,0], return_distance=True)
        dist_on_skel = distance * skel # h,w
        dist_on_skel = dist_on_skel.astype(np.uint8)*255
        # chose the largest connected region, to remove the lone points.
        num_labels, regions, stats, centroids = cv2.connectedComponentsWithStats(dist_on_skel, connectivity=8) # regions H*W, value denotes the i-th region.
        clean_skel = None
        largest_area = 0
        for region_id  in np.unique(regions): # skip the background
            if region_id == 0:
                continue
            curr_skel = (regions == region_id)
            if np.sum(curr_skel) >=largest_area:
                largest_area = np.sum(curr_skel)
                clean_skel = curr_skel
        if clean_skel is not None:
            skeleton_canvas[clean_skel] = 255
            max_depth_path = get_dfs_points(clean_skel.astype(np.int32))
            # scribble_points = get_bezier_points(np.array(max_depth_path))
            scribble_points = np.array(max_depth_path)
            scribble_points = scribble_points.astype(np.int32)
            scribble_canvas = cv2.polylines(scribble_canvas,scribble_points[None,:,:],False,[curr_category,curr_category,curr_category],2)
            scribble_canvas_color[scribble_canvas[:,:,0] == curr_category] = colormap[curr_category]
        
        
        img_gt_vis = np.concatenate((img,curr_mask_color,scribble_canvas_color),axis=0)

        cv2.imwrite(gt_path,curr_mask)
        cv2.imwrite(gt_vis_path,img_gt_vis)
        cv2.imwrite(scribble_path,scribble_canvas)
        cv2.imwrite(scribble_vis_path,scribble_canvas_color)
        
                
        cv2.imwrite(os.path.join(vis_skeleton_outputdir,filename),skeleton_canvas)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir',default='./COCO2014',type=str)
    parser.add_argument('--dataType',default='train2014',type=str)
    parser.add_argument('--save_dir',default='./coco2014_train_scribble_random',type=str)
    parser.add_argument('--random',default='True',type=str,help='The generate the scribble with a random walk path, or chose the longest path as the scribble')
    parser.add_argument('--numworkers',default=1,type=int)
    args=parser.parse_args()
    print(args)
    
    
    dataDir = args.dataDir
    dataType= args.dataType
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    saveDir = args.save_dir
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    numworkers = args.numworkers
    # initialize COCO api for instance annotations
    coco=COCO(annFile)
    catIds = coco.getCatIds()
    ImgIDs = coco.getImgIds()
    AnnIDs = coco.getAnnIds()
    print("catIds len:{}, ImgIDs len:{}, AnnIDs:{}".format(len(catIds), len(ImgIDs),len(AnnIDs)))

    print(catIds)
    category_mapping = {}
    categories = np.arange(0,80)+1
    print(categories)
    for id,cate_id in enumerate(catIds):
        category_mapping[cate_id] = categories[id]
    print(category_mapping)

        
    imginfos = coco.loadImgs(ImgIDs)
    imganns = coco.getAnnIds()
    print(len(imginfos))
    
    uncomplete_imginfos = []
    f = open('uncomplete_list.txt','w')
    for one_info in tqdm(imginfos,ncols=70,total=len(imginfos),desc='Checking...'): # to skip the completed ones
        filename = one_info['file_name']
        filepath = os.path.join(saveDir,'semantic_gt_vis',filename)
        if os.path.isfile(filepath):
            continue
        else:
            uncomplete_imginfos.append(one_info)
            print(one_info['file_name'],len(coco.getAnnIds(imgIds=one_info['id'])),file=f)
            if len(coco.getAnnIds(imgIds=one_info['id'])) == 0:
                continue
    f.close()
    # imgs2scribble(imginfos)
    
    workers = numworkers
    task_split = []
    split_batchnum = int(len(uncomplete_imginfos)/workers)
    for i in range(workers):
        if i < workers-1:
            current_slice = uncomplete_imginfos[i*split_batchnum:split_batchnum*(i+1)]
            task_split.append(current_slice)
        else:
            current_slice = uncomplete_imginfos[i*split_batchnum:]
            task_split.append(current_slice)
    
    p = Pool(workers)
    for taskid,task_imginfo in enumerate(task_split):
        # imgs2scribble(task_imginfo,taskid)
        p.apply_async(imgs2scribble,(args,task_split[taskid],dataDir,dataType,saveDir,taskid))
    p.close()
    p.join()