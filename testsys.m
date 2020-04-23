% Face Recognition System
% Version : 1.0
% Date : 28.5.2012
% Author : Omid Sakhi

function recognition_rate = testsys(myDatabase,params)

if (params.trained==0)
    fprintf('System is not trained. Please train your system first.\n');
    return;
end

total = 0;
recognition_rate = 0;
fprintf('Please Wait...\n');
data_folder_contents = dir ('./data');
number_of_folders_in_data_folder = size(data_folder_contents,1);
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
    fprintf([person_name,'\n']);
    person_folder_contents = dir(['./data/',person_name,'/*.pgm']);
    number_of_faces_in_person_folder = size(person_folder_contents,1);    
    if (number_of_faces_in_person_folder==10)
        ufft = params.used_faces_for_testing;
    else
        ufft = 1:number_of_faces_in_person_folder;        
    end    
    for face_index=1:size(ufft,2)
        total = total + 1;
        filename = ['./data/',person_name,'/',person_folder_contents(ufft(face_index),1).name];        
        answer_person_index = facerec(filename,myDatabase,params,1);
        if (answer_person_index == person_index)
            recognition_rate = recognition_rate + 1;
        end        
    end
end
recognition_rate = recognition_rate/total*100;
fprintf(['\nRecognition Rate is ',num2str(recognition_rate),'%% for a total of ',num2str(total),' unseen faces.\n']);