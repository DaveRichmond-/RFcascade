%% troubleshoot LK tracking

[imageStack,sizeC,sizeZ,sizeT] = bf_openStack(image_fname);
load(data_fname);

[gbar, R, Psi, Lambda, PsiT, X, Y, CC] = make_appearance_model(imageStack, model_data, xbar_vector, [0:7]);
