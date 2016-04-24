[N I] = size(arcene_train_data);
[N O] = size(arcene_train_labels);


Ntrneq = N*O;
Hub  = -1 + ceil((Ntrneq-O)/(I+O+1));
Hmax = 