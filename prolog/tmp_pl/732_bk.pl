:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.9633957510100117e-08).
nn('X0',1,1.0).
nn('X0',2,5.94841675916058e-11).
nn('X0',3,2.3525784024896706e-20).
nn('X0',4,2.3295566864423023e-12).
nn('X0',5,3.8688874326453515e-10).
nn('X0',6,1.1844085434642437e-10).
nn('X0',7,1.5890423699094214e-10).
nn('X0',8,2.0951336911038954e-13).
nn('X0',9,8.575486783353459e-13).
nn('X1',0,1.0).
nn('X1',1,2.1145113654821857e-14).
nn('X1',2,9.720525166040384e-13).
nn('X1',3,5.654994449683662e-17).
nn('X1',4,6.807872641283208e-22).
nn('X1',5,1.339131722177811e-10).
nn('X1',6,1.966628465979725e-12).
nn('X1',7,1.4679879534185147e-10).
nn('X1',8,7.317548733320933e-17).
nn('X1',9,2.116376173366526e-17).
nn('X2',0,0.0001425930968252942).
nn('X2',1,1.4284345525084063e-05).
nn('X2',2,0.9996439814567566).
nn('X2',3,1.580447595017631e-08).
nn('X2',4,1.112631525757024e-05).
nn('X2',5,1.7037830275512533e-06).
nn('X2',6,9.049495929502882e-06).
nn('X2',7,5.331949068931863e-05).
nn('X2',8,9.243213571608067e-05).
nn('X2',9,3.1537620088784024e-05).
nn('X3',0,9.98310383693024e-07).
nn('X3',1,3.928485341475607e-07).
nn('X3',2,4.8273918764607515e-06).
nn('X3',3,8.346904905920383e-06).
nn('X3',4,3.916406683401874e-07).
nn('X3',5,0.00016106452676467597).
nn('X3',6,1.545409986647428e-06).
nn('X3',7,0.0017110012704506516).
nn('X3',8,0.9975032210350037).
nn('X3',9,0.0006082051550038159).
nn('X4',0,2.442112267617147e-13).
nn('X4',1,4.844010959966709e-16).
nn('X4',2,7.706882795314941e-16).
nn('X4',3,1.4294521400418672e-17).
nn('X4',4,1.0198068080331048e-19).
nn('X4',5,3.660711396014961e-14).
nn('X4',6,1.663585572886031e-24).
nn('X4',7,1.0).
nn('X4',8,2.6916529833896546e-20).
nn('X4',9,6.318269796246057e-10).
nn('X5',0,7.464632346488101e-14).
nn('X5',1,3.268552442592387e-11).
nn('X5',2,1.9516816607278997e-08).
nn('X5',3,6.611747815910007e-10).
nn('X5',4,1.2831938289714628e-10).
nn('X5',5,3.414614813124217e-09).
nn('X5',6,1.821258074907739e-11).
nn('X5',7,6.675652002741117e-06).
nn('X5',8,0.999981164932251).
nn('X5',9,1.2197569958516397e-05).

a :- Pos=[f(['X0','X1','X2','X3'],11),f(['X4','X5'],15)], metaabd(Pos).
