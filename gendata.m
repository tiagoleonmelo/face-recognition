% Face Recognition System
% Version : 1.0
% Date : 28.5.2012
% Author : Omid Sakhi

function [myDatabase,params] = gendata(params)
% This module reads all the folders inside data folder. Each folder belongs
% to one person and the system assumes that the name of the person is equal
% to the name of its folder.
% In each folder there are 10 images of one person. The system use 5 of
% those images to generate its database. If there are less than 5 images
% inside a folder, the system use all of them for training. Even if there
% is one face photo.

% these parameters can be changed with care
params.blk_height = 5;
params.blk_overlap = 4;
params.coeff1_quant = 18;
params.coeff2_quant = 10;
params.coeff3_quant = 7;
params.number_of_states = 7;
params.face_height = 56;
params.face_width = 46;
params.used_faces_for_training = [1 5 6 8 10];
params.used_faces_for_testing = [2 3 4 7 9];
%

params.eps=.000001;
params.trained = 0;
fprintf ('Loading Faces ...\n');
params.number_of_blocks = (params.face_height-params.blk_height)/(params.blk_height-params.blk_overlap)+1;
data_folder_contents = dir ('./data');
number_of_folders_in_data_folder = size(data_folder_contents,1);
myDatabase = cell(0,0);
person_index = 0;
for person=1:number_of_folders_in_data_folder
    if strcmp(data_folder_contents(person,1).name,'.') % is not a folder -> skip
        continue;
    end
    if strcmp(data_folder_contents(person,1).name,'..') % is not a folder -> skip
        continue;
    end
    if (data_folder_contents(person,1).isdir == 0) % is a file -> skip
        continue;
    end
    person_index = person_index+1;
    person_name = data_folder_contents(person,1).name;
    myDatabase{1,person_index} = person_name;
    fprintf([person_name,' ']);
    person_folder_contents = dir(['./data/',person_name,'/*.pgm']);
    number_of_images_in_person_folder = size(person_folder_contents,1);
    blk_cell = cell(0,0);
    if (number_of_images_in_person_folder==10)
        ufft = params.used_faces_for_training;
    else
        ufft = 1:number_of_images_in_person_folder;
    end
    number_of_faces_for_train = size(ufft,2);
    for face_index=1:number_of_faces_for_train
        I = imread(['./data/',person_name,'/',person_folder_contents(ufft(face_index),1).name]);
        try
            I = rgb2gray(I);
        end
        I = imresize(I,[params.face_height params.face_width]);
        I = ordfilt2(I,1,true(3));
        blk_begin = 1;
        blk_index = 0;
        for blk_end=params.blk_height:params.blk_height-params.blk_overlap:params.face_height
            blk = I(blk_begin:blk_end,:);
            blk_double = double(blk);
            [U,S,V] = svd(blk_double);
            coeff1 = U(1,1);
            coeff2 = S(1,1);
            coeff3 = S(2,2);
            blk_index=blk_index+1;
            blk_cell{blk_index,face_index} = [coeff1 coeff2 coeff3];
            blk_begin = blk_begin + (params.blk_height-params.blk_overlap);
        end
    end
    myDatabase{2,person_index} = blk_cell;
    if (mod(person_index,10)==0) %after 10 rows, move to a newline for display
        fprintf('\n');
    end
end
coeff1 = [];
coeff2 = [];
coeff3 = [];
n_persons = size(myDatabase,2);
for person_index=1:n_persons
    [n_blocks,n_images] = size(myDatabase{2,person_index});
    for image_index=1:n_images
        for block_index=1:n_blocks
            if (isempty(myDatabase{2,person_index}{block_index,image_index}))
                continue;
            end
            coeff1(:,end+1) = myDatabase{2,person_index}{block_index,image_index}(1,1);
            coeff2(:,end+1) = myDatabase{2,person_index}{block_index,image_index}(1,2);
            coeff3(:,end+1) = myDatabase{2,person_index}{block_index,image_index}(1,3);
        end
    end
end
max_coeff1 = max(coeff1(:));
max_coeff2 = max(coeff2(:));
max_coeff3 = max(coeff3(:));
min_coeff1 = min(coeff1(:));
min_coeff2 = 0;%min(coeff2(:)); %it works better with zero
min_coeff3 = 0;%min(coeff3(:)); %it works better with zero
%delta is the width of each bin for quantization
delta_coeff1 = (max_coeff1-min_coeff1)/(params.coeff1_quant-params.eps);
delta_coeff2 = (max_coeff2-min_coeff2)/(params.coeff2_quant-params.eps);
delta_coeff3 = (max_coeff3-min_coeff3)/(params.coeff3_quant-params.eps);
params.coeff_stats = [min_coeff1 min_coeff2 min_coeff3;...
    max_coeff1 max_coeff2 max_coeff3;...
    delta_coeff1 delta_coeff2 delta_coeff3];
min_label = Inf;
max_label = -Inf;
for person_index=1:n_persons
    for image_index=1:n_images
        for block_index=1:n_blocks
            if (isempty(myDatabase{2,person_index}{block_index,image_index}))
                continue;
            end
            blk_coeffs = myDatabase{2,person_index}{block_index,image_index};
            min_coeffs = params.coeff_stats(1,:);
            delta_coeffs = params.coeff_stats(3,:);
            qt = floor((blk_coeffs-min_coeffs)./delta_coeffs);
            myDatabase{3,person_index}{block_index,image_index} = qt;
            label = qt(1)* params.coeff2_quant*params.coeff3_quant + qt(2) * params.coeff3_quant + qt(3)+1;            
            min_label = min([label min_label]);
            max_label = max([label max_label]);
            myDatabase{4,person_index}{block_index,image_index} = label;
        end
        myDatabase{5,person_index}{1,image_index} = cell2mat(myDatabase{4,person_index}(:,image_index));
    end
end
params.min_label = min_label; %for information only. We don't use it later
params.max_label = max_label; %for information only. We don't use it later
fprintf('done.\n');
save DATABASE myDatabase params;