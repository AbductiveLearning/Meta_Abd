:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.4498945094819646e-09).
nn('X0',1,1.0).
nn('X0',2,6.567715121752471e-11).
nn('X0',3,2.2530206726410215e-21).
nn('X0',4,4.7880922588328795e-14).
nn('X0',5,2.1798591577637527e-12).
nn('X0',6,1.5394014730475691e-13).
nn('X0',7,2.1865656507635833e-10).
nn('X0',8,2.1352730747556305e-15).
nn('X0',9,1.0002675601597077e-14).
nn('X1',0,9.517058148500457e-12).
nn('X1',1,6.118465976923115e-12).
nn('X1',2,5.606792910839431e-06).
nn('X1',3,6.992224926335513e-20).
nn('X1',4,0.9999896287918091).
nn('X1',5,2.2511092083732365e-06).
nn('X1',6,2.329634980924311e-06).
nn('X1',7,4.887605251080451e-11).
nn('X1',8,1.0136927782638042e-14).
nn('X1',9,1.1754286077803044e-07).
nn('X2',0,1.0).
nn('X2',1,4.513392295434082e-16).
nn('X2',2,1.8391609346579685e-09).
nn('X2',3,8.457894631830616e-17).
nn('X2',4,5.554458359687054e-20).
nn('X2',5,4.0391697148436934e-13).
nn('X2',6,1.2981202484468879e-12).
nn('X2',7,1.1659453784396145e-11).
nn('X2',8,3.1075104512182095e-14).
nn('X2',9,9.88003285413118e-15).
nn('X3',0,1.0).
nn('X3',1,2.017005789718129e-23).
nn('X3',2,2.719429262126264e-14).
nn('X3',3,1.6649606591329322e-27).
nn('X3',4,5.195628092626893e-34).
nn('X3',5,3.195840872296053e-22).
nn('X3',6,3.4330770999646676e-20).
nn('X3',7,1.913370861942015e-19).
nn('X3',8,4.832916498360815e-24).
nn('X3',9,4.3118529656483485e-24).
nn('X4',0,2.417249902464391e-07).
nn('X4',1,2.0096813482025494e-13).
nn('X4',2,2.6192674340563826e-07).
nn('X4',3,8.696444917812539e-15).
nn('X4',4,8.841550879878923e-06).
nn('X4',5,9.577534910931718e-06).
nn('X4',6,0.999981164932251).
nn('X4',7,1.6456448833843118e-12).
nn('X4',8,1.6305617966949626e-10).
nn('X4',9,3.4834580966286444e-12).

a :- Pos=[f(['X0','X1','X2'],5),f(['X3','X4'],6)], metaabd(Pos).
