% Face Recognition System
% Version : 1.0
% Date : 28.5.2012
% Author : Omid Sakhi

function [person_index,maxlogpseq] = facerec(filename,myDatabase,params,display_filename)

if (params.trained==0)
    fprintf('System is not trained. Please train your system first.\n');
    return;
end
I = imread(filename);
try
    I = rgb2gray(I);                
end
I = imresize(I,[params.face_height params.face_width]);
I = ordfilt2(I,1,true(3));
blk_begin = 1;
blk_index = 0;        
n_blocks = 0;
blk_cell = cell((params.face_height-params.blk_height)/(params.blk_height-params.blk_overlap)+1,1);
% generate features from image blocks
for blk_end=params.blk_height:params.blk_height-params.blk_overlap:params.face_height
    n_blocks = n_blocks + 1;
    blk = I(blk_begin:blk_end,:);
    blk_double = double(blk);
    [U,S,V] = svd(blk_double);
    coeff1 = U(1,1);
    coeff2 = S(1,1);
    coeff3 = S(2,2);    
    blk_index=blk_index+1;
    blk_cell{blk_index,1} = [coeff1 coeff2 coeff3];
    blk_begin = blk_begin + (params.blk_height-params.blk_overlap);
end     

%quantize features of each block, assign label to each and generate sequence
seq = zeros(1,n_blocks);
for block_index=1:n_blocks
        blk_coeffs = blk_cell{block_index,1};
        min_coeffs = params.coeff_stats(1,:);
        max_coeffs = params.coeff_stats(2,:);
        blk_coeffs = max([blk_coeffs;min_coeffs]);        
        blk_coeffs = min([blk_coeffs;max_coeffs]);                
        delta_coeffs = params.coeff_stats(3,:);
        qt = floor((blk_coeffs-min_coeffs)./delta_coeffs);
        label = qt(1)* params.coeff2_quant*params.coeff3_quant + qt(2) * params.coeff3_quant + qt(3)+1;               
        seq(1,block_index) = label;
end

%calculate sequence probability for each face model in the database
number_of_persons_in_database = size(myDatabase,2);
results = zeros(1,number_of_persons_in_database);
for i=1:number_of_persons_in_database    
    TRANS = myDatabase{6,i}{1,1}; %transition probabilities of the trained model
    EMIS = myDatabase{6,i}{1,2}; %emission probabilities of the trained model
    [ignore,logpseq] = hmmdecode(seq,TRANS,EMIS);    
    P=exp(logpseq);
    results(1,i) = P;
end
[maxlogpseq,person_index] = max(results);
if (display_filename) 
    fprintf(['The person ',filename,' whom you are looking for is ',myDatabase{1,person_index},'.\n']);
else
	fprintf(['The person whom you are looking for is ',myDatabase{1,person_index},'.\n']);    
end