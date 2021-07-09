%% script to test dataPreprocessing
%% created by Fu-Jen Chu on 09/15/2016

close all
clear

%parpool(4)
load cls2img

% generate list for splits
list = [26:255 268:281];
list_idx = randperm(length(list));
train_list_idx = list_idx(floor(length(list)/5)+1:end);
test_list_idx = list_idx(1:floor(length(list)/5));
train_list_cls = list(train_list_idx);
test_list_cls = list(test_list_idx);

train_list = [];
for idx = 1:length(train_list_cls)
    train_list = [train_list cls2img{train_list_cls(idx)}];
end

test_list = [];
for idx = 1:length(test_list_cls)
    test_list = [test_list cls2img{test_list_cls(idx)}];
end


for folder = 1:10
display(['processing folder ' int2str(folder)])

imgDataDir = ['/home/fujenchu/projects/deepLearning/tensorflow-finetune-flickr-style-master/data/grasps/' sprintf('%02d',folder)];
imgDepthDataDir = ['/home/fujenchu/projects/deepLearning/tensorflow-finetune-flickr-style-master/data/grasps/' sprintf('%02d',folder) '_rgd'];
txtDataDir = ['/home/fujenchu/projects/deepLearning/tensorflow-finetune-flickr-style-master/data/grasps/' sprintf('%02d',folder)];

imgDataOutDir = '/media/fujenchu/data/fasterrcnn_grasp/res50_rgbd_grasp_5_5_5_object_tf/data/Images';
imgDepthDataOutDir = '/media/fujenchu/data/fasterrcnn_grasp/res50_rgbd_grasp_5_5_5_object_tf/data/ImagesDepth';
labelSetTrain = '/media/fujenchu/data/fasterrcnn_grasp/res50_rgbd_grasp_5_5_5_object_tf/data/train_label.txt';
labelSetTest = '/media/fujenchu/data/fasterrcnn_grasp/res50_rgbd_grasp_5_5_5_object_tf/data/test_label.txt';
imgSetTrain = '/media/fujenchu/data/fasterrcnn_grasp/res50_rgbd_grasp_5_5_5_object_tf/data/train_img.txt'; 
imgSetTest = '/media/fujenchu/data/fasterrcnn_grasp/res50_rgbd_grasp_5_5_5_object_tf/data/test_img.txt'; 
imgSetTestfull = '/media/fujenchu/data/fasterrcnn_grasp/res50_rgbd_grasp_5_5_5_object_tf/data/testfull.txt'; 

imgFiles = dir([imgDataDir '/*.png']);
imgDepthFiles = dir([imgDepthDataDir '/*.png']);
txtFiles = dir([txtDataDir '/*pos.txt']);

logfileID = fopen('log.txt','a');
%mainfileID = fopen(['/home/fujenchu/projects/deepLearning/deepGraspExtensiveOffline/data/grasps/scripts/trainttt' sprintf('%02d',folder) '.txt'],'a');
for idx = 1:length(imgFiles) 
    %% display progress
    tic
    display(['processing folder: ' sprintf('%02d',folder) ', imgFiles: ' int2str(idx)])
    
    %% reading data
    imgName = imgFiles(idx).name;
    [pathstr,imgname] = fileparts(imgName);
    
 
    
    rotatePara = 5;
    shitfPara = 5;
    imgSet = imgSetTrain;
    labelSet = labelSetTrain;
    filenum = str2num(imgname(4:7));
    if(any(test_list == filenum))
        file_writeID = fopen(imgSetTestfull,'a');
        fprintf(file_writeID, '%s\n', [imgDataDir '_Cropped320_rgd/' imgname '_rgd_preprocessed_1.png' ] );
        fclose(file_writeID);
        
        rotatePara = 1;
        shitfPara = 1;
        imgSet = imgSetTest;
        labelSet = labelSetTest;
    end
    
    
    txtName = txtFiles(idx).name;
    [pathstr,txtname] = fileparts(txtName);

    img = imread([imgDataDir '/' imgname '.png']);
    imgDepth = imread([imgDepthDataDir '/' imgname '_rgd.png']);

    fileID = fopen([txtDataDir '/' txtname '.txt'],'r');
    sizeA = [2 100];
    bbsIn_all = fscanf(fileID, '%f %f', sizeA);
    fclose(fileID);
    
    %% data pre-processing
    [imagesOut_depth bbsOut] = dataPreprocessing_fasterrcnnD(img, imgDepth, bbsIn_all, 224, rotatePara, shitfPara);
    
    % for each augmented image
    labelfile_writeID = fopen(labelSet,'a');
    imgfile_writeID = fopen(imgSet,'a');
    for i = 1:1:size(imagesOut_depth,2)
        
        % for each bbs
        printCount = 0;
        if size(bbsOut{i},2) == 0
            continue
        end
        for ibbs = 1:1:size(bbsOut{i},2)
          A = bbsOut{i}{ibbs};  
          xy_ctr = sum(A,2)/4; x_ctr = xy_ctr(1); y_ctr = xy_ctr(2);
          width = sqrt(sum((A(:,1) - A(:,2)).^2)); height = sqrt(sum((A(:,2) - A(:,3)).^2));
          if(A(1,1) > A(1,2))
              theta = atan((A(2,2)-A(2,1))/(A(1,1)-A(1,2)));
          else
              theta = atan((A(2,1)-A(2,2))/(A(1,2)-A(1,1))); % note y is facing down
          end  
    
          % process to fasterrcnn
          x_min = x_ctr - width/2; x_max = x_ctr + width/2;
          y_min = y_ctr - height/2; y_max = y_ctr + height/2;
          %if(x_min < 0 || y_min < 0 || x_max > 227 || y_max > 227) display('yoooooooo'); end
          if((x_min < 0 && x_max < 0) || (y_min > 224 && y_max > 224) || (x_min > 224 && x_max > 224) || (y_min < 0 && y_max < 0)) display('xxxxxxxxx'); continue; end
          cls = round((theta/pi*180+90)/10) + 1;
          
          % write as lefttop rightdown, Xmin Ymin Xmax Ymax, ex: 261 109 511 705  (x水平 y垂直)
          %fprintf(labelfile_writeID, '%d %f %f %f %f\n', cls, x_min, y_min, x_max, y_max );  
          fprintf(labelfile_writeID, '%d %f %f %f %f\n', cls, x_ctr, y_ctr, width, height); 
          fprintf(imgfile_writeID, '%s\n', [imgname '_preprocessed_' int2str(i) '_' int2str(ibbs) '.png'] );
          %imwrite(imagesOut{i}, [imgDataOutDir '/' imgname '_preprocessed_' int2str(i) '_' int2str(ibbs) '.png']); 
          %img_ddd = imagesOut_depth{i};
          %img_ddd(:,:,1) = img_ddd(:,:,3);
          %img_ddd(:,:,2) = img_ddd(:,:,3);
          imwrite(imagesOut_depth{i}, [imgDepthDataOutDir '/' imgname '_preprocessed_' int2str(i) '_' int2str(ibbs) '.png']); 

          printCount = printCount+1;
        end

    end
    fclose(labelfile_writeID);
    fclose(imgfile_writeID);

    toc
end

end
