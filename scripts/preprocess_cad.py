import os

# modify the path where your dataset stored
test_data_folder = "/path/of/CamouflagedAnimalDataset/"
gt_dir = "/path/of/new_gt/"

classes = os.listdir(test_data_folder)
classes = [cls for cls in classes if cls[-4:] != ".txt" and cls[-4:] != "json" and cls[0] != '.']
for cls in classes:
    new_gt_dir = gt_dir + cls + '/groundtruth/'
    gt_files = [
        p for p in os.listdir(new_gt_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]

    for old_name in gt_files:
        new_name = old_name.split('_')[0]+'.png'
        os.system("mv %s %s "%(os.path.join(new_gt_dir, old_name), os.path.join(new_gt_dir, new_name)))

    video_dir = test_data_folder + cls + '/frames/'
    new_video_dir = test_data_folder + cls + '/new_frames'

    os.mkdir(os.path.join(new_video_dir))
    for old_name in gt_files:
        old_name = old_name.split('_')[0]+'.png'
        new_frame = cls + '_' + old_name
        os.system("cp %s %s "%(os.path.join(video_dir, new_frame), os.path.join(new_video_dir, old_name)))