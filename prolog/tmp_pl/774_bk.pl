:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.4263175601540723e-13).
nn('X0',1,2.6892171639086556e-15).
nn('X0',2,1.0799527504906647e-12).
nn('X0',3,8.578617638477226e-08).
nn('X0',4,3.107056727458257e-07).
nn('X0',5,8.330216516583278e-09).
nn('X0',6,1.2682137275293968e-16).
nn('X0',7,9.824563312577084e-05).
nn('X0',8,8.152389874283017e-10).
nn('X0',9,0.999901294708252).
nn('X1',0,4.472137334232684e-06).
nn('X1',1,0.0008031624020077288).
nn('X1',2,0.03469936177134514).
nn('X1',3,0.021374717354774475).
nn('X1',4,0.006693725474178791).
nn('X1',5,0.0026778297033160925).
nn('X1',6,9.55896102823317e-05).
nn('X1',7,0.0048451609909534454).
nn('X1',8,0.9142088294029236).
nn('X1',9,0.014597257599234581).
nn('X2',0,3.748933607994331e-09).
nn('X2',1,7.417231245199218e-06).
nn('X2',2,0.00017379737982992083).
nn('X2',3,0.0031805036123842).
nn('X2',4,0.0003647002449724823).
nn('X2',5,0.00046328402822837234).
nn('X2',6,5.201075126137766e-10).
nn('X2',7,0.9705260396003723).
nn('X2',8,7.777236419315159e-07).
nn('X2',9,0.025283513590693474).
nn('X3',0,1.0046199427904412e-09).
nn('X3',1,3.641964212874882e-06).
nn('X3',2,4.677081960835494e-05).
nn('X3',3,7.217622055577522e-09).
nn('X3',4,1.3628442374624683e-08).
nn('X3',5,1.8150412870454602e-06).
nn('X3',6,8.887297440196562e-07).
nn('X3',7,8.382203304790892e-06).
nn('X3',8,0.9999372959136963).
nn('X3',9,1.1870819207615568e-06).

a :- Pos=[f(['X0','X1','X2','X3'],32)], metaabd(Pos).