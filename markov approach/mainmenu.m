% Face Recognition System
% Version : 1.0
% Date : 28.5.2012
% Author : Omid Sakhi
% Original Paper : 
%   H. Miar-Naimi and P. Davari A New Fast and Efficient HMM-Based 
%   Face Recognition System Using a 7-State HMM Along With SVD Coefficients

clear all;
close all;
clc;

if (exist('DATABASE.mat','file'))
    load DATABASE.mat;
end
while (1==1)
    choice=menu('Face Recognition',...
                'Generate Database',...  
                'Load Database',...
                'Train System',...
                'Calculate Recognition Rate',...
                'Recognize One Person',...
                'Exit');
    if (choice ==1)
        if (~exist('DATABASE.mat','file'))
            [myDatabase params] = gendata();        
        else
            pause(0.001);    
            choice2 = questdlg('Generating a new database will remove any previous trained database. Are you sure?', ...
                               'Warning...',...
                               'Yes', ...
                               'No','No');            
            switch choice2
                case 'Yes'
                    pause(0.1);
                    [myDatabase params] = gendata();        
                case 'No'
            end
        end        
    end
    if (choice == 2)
        if (~exist('DATABASE.mat','file'))
            fprintf('Database file does not exist. Please generate it first!\n');
        else
            load DATABASE.mat;
            fprintf('Database is now loaded.\n');
        end
    end    
    if (choice == 3)
        if (~exist('myDatabase','var'))
            fprintf('Please generate or load database first!\n');
        else
             if (params.trained==0)
                 [myDatabase params] = trainsys(myDatabase,params);                
             else
                pause(0.001);
                choice2 = questdlg('Your database is already trained. Do you really want to re-train your data?', ...
                                   'Warning...',...
                                   'Yes', ...
                                   'No','No');            
                switch choice2
                    case 'Yes'                     
                         pause(0.1);
                         [myDatabase params] = trainsys(myDatabase,params);                
                    case 'No'
                end                        
             end
        end                
    end    
    if (choice == 4)
        if (~exist('myDatabase','var'))
            fprintf('Please generate or load database first!\n');
        else
            recognition_rate = testsys(myDatabase,params);                
        end                        
    end    
    if (choice == 5)
        if (~exist('myDatabase','var'))
            fprintf('Please load database first!\n');
        else            
            pause(0.001);
            % show file dialog box and get one image from the user
            [file_name file_path] = uigetfile ({'*.pgm';'*.jpg';'*.png'});
            if file_path ~= 0
                filename = [file_path,file_name];                
                facerec (filename,myDatabase,params,0);                        
            end
        end
    end
    if (choice == 6)
        clear choice choice2
        return;
    end    
end