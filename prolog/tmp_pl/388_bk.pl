:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.8191596609540284e-05).
nn('X0',1,1.6394199775504603e-08).
nn('X0',2,1.6133492408698658e-06).
nn('X0',3,2.1032289798661452e-10).
nn('X0',4,1.0131716408068314e-05).
nn('X0',5,0.001719441032037139).
nn('X0',6,0.9982502460479736).
nn('X0',7,1.7907101357295119e-09).
nn('X0',8,3.9992269762478827e-07).
nn('X0',9,1.4561063466089763e-09).
nn('X1',0,3.410029550399152e-11).
nn('X1',1,6.598485242648167e-07).
nn('X1',2,4.151243410888128e-05).
nn('X1',3,0.9998894929885864).
nn('X1',4,3.321354996588255e-11).
nn('X1',5,3.574236325221136e-05).
nn('X1',6,6.036437219221827e-14).
nn('X1',7,3.251271118642762e-05).
nn('X1',8,1.6801619651118926e-08).
nn('X1',9,2.7708054517461278e-08).
nn('X2',0,1.6166437077913542e-08).
nn('X2',1,2.4263635545196394e-09).
nn('X2',2,2.431483210330043e-07).
nn('X2',3,3.550245310179889e-05).
nn('X2',4,2.8733509225276066e-06).
nn('X2',5,2.854459921763919e-07).
nn('X2',6,3.358793495070178e-12).
nn('X2',7,0.15920978784561157).
nn('X2',8,2.9184991490183165e-06).
nn('X2',9,0.8407483696937561).
nn('X3',0,8.212342805791195e-08).
nn('X3',1,0.0002490807673893869).
nn('X3',2,0.0025668886955827475).
nn('X3',3,0.9964866638183594).
nn('X3',4,1.1084553275253484e-09).
nn('X3',5,0.00025122499209828675).
nn('X3',6,2.190100878429746e-10).
nn('X3',7,0.0004431511915754527).
nn('X3',8,2.728704430410289e-06).
nn('X3',9,2.004121739673792e-07).
nn('X4',0,0.0005137619446031749).
nn('X4',1,0.9955234527587891).
nn('X4',2,0.0008327739196829498).
nn('X4',3,9.9742210295517e-06).
nn('X4',4,4.0783263102639467e-05).
nn('X4',5,2.9747747248620726e-05).
nn('X4',6,3.596977649067412e-06).
nn('X4',7,0.002671686001121998).
nn('X4',8,6.329542884486727e-06).
nn('X4',9,0.00036774398176930845).
nn('X5',0,2.1523922555388708e-07).
nn('X5',1,8.973972259740393e-11).
nn('X5',2,9.645479440223426e-05).
nn('X5',3,5.57259137880562e-17).
nn('X5',4,0.9991165995597839).
nn('X5',5,0.00040951944538392127).
nn('X5',6,0.00037475471617653966).
nn('X5',7,3.693367833790262e-09).
nn('X5',8,5.308148018356995e-11).
nn('X5',9,2.4684868549229577e-06).

a :- Pos=[f(['X0','X1','X2','X3'],21),f(['X4','X5'],5)], metaabd(Pos).
