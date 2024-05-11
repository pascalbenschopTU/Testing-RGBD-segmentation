call convert_and_train_new.bat . SUNRGBD . 60 3 DeepLab

cd ..

@REM Train another time
call convert_and_train_new.bat . SUNRGBD . 60 3 DFormer_Large

@REM shutdown /h