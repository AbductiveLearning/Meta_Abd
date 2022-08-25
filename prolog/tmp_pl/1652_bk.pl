:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.00010643745918059722).
nn('X0',1,0.9998229146003723).
nn('X0',2,3.461434971541166e-05).
nn('X0',3,1.747098465898489e-08).
nn('X0',4,2.001159737119451e-05).
nn('X0',5,1.237310357282695e-06).
nn('X0',6,5.3252688303473406e-06).
nn('X0',7,4.676517619373044e-06).
nn('X0',8,2.212441813753685e-06).
nn('X0',9,2.624251919769449e-06).
nn('X1',0,0.9999998807907104).
nn('X1',1,9.315418309776766e-14).
nn('X1',2,5.216758580672831e-08).
nn('X1',3,7.311205460491799e-12).
nn('X1',4,5.585068015299998e-15).
nn('X1',5,6.368104377152406e-10).
nn('X1',6,8.765509984520747e-10).
nn('X1',7,6.086713000286181e-08).
nn('X1',8,1.1126032539721109e-08).
nn('X1',9,4.4500989004880864e-10).
nn('X2',0,5.09787367874992e-09).
nn('X2',1,1.0).
nn('X2',2,1.66531177736573e-10).
nn('X2',3,2.6008915923412966e-19).
nn('X2',4,8.465465199496147e-13).
nn('X2',5,5.216322512824334e-11).
nn('X2',6,1.2010276670762488e-12).
nn('X2',7,1.1883289019465337e-08).
nn('X2',8,5.5572940590694134e-14).
nn('X2',9,7.081463236982555e-13).
nn('X3',0,5.442841056330083e-12).
nn('X3',1,8.646887893204447e-12).
nn('X3',2,5.754471552044116e-11).
nn('X3',3,4.79011214338243e-06).
nn('X3',4,1.8955111954710446e-05).
nn('X3',5,1.0867314585993881e-06).
nn('X3',6,1.165930179280409e-14).
nn('X3',7,0.00212184083648026).
nn('X3',8,6.640949834491039e-08).
nn('X3',9,0.9978532195091248).
nn('X4',0,0.0008726372616365552).
nn('X4',1,0.038430530577898026).
nn('X4',2,0.011995823122560978).
nn('X4',3,0.0019255760125815868).
nn('X4',4,0.3620033860206604).
nn('X4',5,0.2546321153640747).
nn('X4',6,0.0005896908114664257).
nn('X4',7,0.06528933346271515).
nn('X4',8,0.0025845414493232965).
nn('X4',9,0.2616763710975647).
nn('X5',0,2.068535772357938e-15).
nn('X5',1,6.111088604555601e-14).
nn('X5',2,1.8067611051719723e-07).
nn('X5',3,6.687043731509156e-25).
nn('X5',4,0.9999993443489075).
nn('X5',5,3.1912145459500607e-07).
nn('X5',6,4.7458530438859725e-09).
nn('X5',7,1.1217218356973126e-14).
nn('X5',8,2.1173532391060737e-17).
nn('X5',9,2.5069877285233133e-09).
nn('X6',0,3.489816080559649e-08).
nn('X6',1,1.9647481477045106e-14).
nn('X6',2,2.125155518228894e-08).
nn('X6',3,3.0411029199227065e-16).
nn('X6',4,4.679760934322985e-08).
nn('X6',5,3.5098574358016776e-07).
nn('X6',6,0.9999995231628418).
nn('X6',7,2.038036995839331e-14).
nn('X6',8,1.2572920135567323e-11).
nn('X6',9,9.281680309861269e-15).

a :- Pos=[f(['X0','X1','X2','X3','X4'],16),f(['X5','X6'],10)], metaabd(Pos).
