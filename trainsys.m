% Face Recognition System
% Version : 1.0
% Date : 28.5.2012
% Author : Omid Sakhi

function [myDatabase params] = trainsys(myDatabase,params)

%parameters
params.trained = 1;
params.number_of_labels = ...
    params.coeff1_quant*...
    params.coeff2_quant*...
    params.coeff3_quant;
%

TRGUESS = ones(params.number_of_states,params.number_of_states) * 0.1;% params.eps;
TRGUESS(params.number_of_states,params.number_of_states) = 1;
for r=1:params.number_of_states-1
    for c=2:params.number_of_states
        TRGUESS(r,c) = 0.5;
        TRGUESS(r,c-1) = 0.5;
    end
end

EMITGUESS = (1/params.number_of_labels)*ones(params.number_of_states,params.number_of_labels);

fprintf('Training ...\n');
n_persons = size(myDatabase,2);
for person_index=1:n_persons
    fprintf([myDatabase{1,person_index},' ']);
    seqmat = cell2mat(myDatabase{5,person_index})';
    [ESTTR,ESTEMIT]=hmmtrain(seqmat,TRGUESS,EMITGUESS,'Tolerance',.01,'Maxiterations',10,'Algorithm', 'BaumWelch');
    ESTTR = max(ESTTR,params.eps);
    ESTEMIT = max(ESTEMIT,params.eps);
    myDatabase{6,person_index}{1,1} = ESTTR;
    myDatabase{6,person_index}{1,2} = ESTEMIT;
    if (mod(person_index,10)==0)
        fprintf('\n');
    end
end
fprintf('done.\n');
save DATABASE myDatabase params