:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.146662361757201e-11).
nn('X0',1,1.590968584501829e-17).
nn('X0',2,2.3290622360415403e-13).
nn('X0',3,7.964493684760034e-22).
nn('X0',4,9.525601065308448e-14).
nn('X0',5,2.295093679549609e-07).
nn('X0',6,0.9999997615814209).
nn('X0',7,1.259735895528698e-18).
nn('X0',8,1.5264599276970138e-14).
nn('X0',9,1.9700369344284455e-20).
nn('X1',0,1.4635706535592874e-10).
nn('X1',1,1.5311360357372905e-06).
nn('X1',2,2.0988989035686245e-06).
nn('X1',3,0.9998431205749512).
nn('X1',4,1.3930874750750673e-14).
nn('X1',5,0.00015316298231482506).
nn('X1',6,2.421529553346116e-15).
nn('X1',7,2.7624890819311076e-08).
nn('X1',8,6.837955982691435e-12).
nn('X1',9,8.657569799952469e-11).
nn('X2',0,8.049259037079537e-09).
nn('X2',1,2.1517395509818016e-07).
nn('X2',2,1.872897155408282e-05).
nn('X2',3,0.9999759197235107).
nn('X2',4,8.795696003407062e-16).
nn('X2',5,5.1938786782557145e-06).
nn('X2',6,1.625636474392178e-14).
nn('X2',7,2.5723775887165345e-10).
nn('X2',8,3.9729165595379923e-13).
nn('X2',9,1.347202107900361e-13).
nn('X3',0,2.9086237418596284e-08).
nn('X3',1,5.066810263087973e-05).
nn('X3',2,0.0002725661324802786).
nn('X3',3,0.9994288086891174).
nn('X3',4,1.2505045610122778e-11).
nn('X3',5,0.0002469810133334249).
nn('X3',6,1.5598984604014987e-10).
nn('X3',7,8.96151334472961e-07).
nn('X3',8,5.160733707043619e-08).
nn('X3',9,1.1781028375068558e-09).
nn('X4',0,0.999998927116394).
nn('X4',1,6.363875398873731e-11).
nn('X4',2,1.045389467435598e-06).
nn('X4',3,3.280137966799046e-11).
nn('X4',4,7.698398532150416e-14).
nn('X4',5,3.3028486612352026e-09).
nn('X4',6,2.225250028686787e-08).
nn('X4',7,3.870498144209478e-09).
nn('X4',8,2.0473109874075135e-09).
nn('X4',9,9.327148775550853e-11).
nn('X5',0,1.3257708299241333e-11).
nn('X5',1,2.7590778620134593e-11).
nn('X5',2,1.1887391515585932e-08).
nn('X5',3,1.7680390484997588e-09).
nn('X5',4,8.980886347098593e-12).
nn('X5',5,1.314697462007075e-09).
nn('X5',6,1.9879536905520467e-10).
nn('X5',7,1.9993444766441826e-06).
nn('X5',8,0.9999935626983643).
nn('X5',9,4.44107354269363e-06).
nn('X6',0,9.931552975850355e-12).
nn('X6',1,3.1413316392558954e-09).
nn('X6',2,3.6150541404822434e-07).
nn('X6',3,6.286721543347085e-08).
nn('X6',4,1.0852886589418631e-08).
nn('X6',5,2.800837428651448e-08).
nn('X6',6,4.922861007727874e-10).
nn('X6',7,5.58429783268366e-05).
nn('X6',8,0.9997805953025818).
nn('X6',9,0.00016310272621922195).

a :- Pos=[f(['X0','X1','X2','X3','X4'],15),f(['X5','X6'],16)], metaabd(Pos).