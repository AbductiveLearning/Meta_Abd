:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,4.579127760721218e-20).
nn('X0',2,4.61777914084055e-13).
nn('X0',3,1.0308152733606071e-22).
nn('X0',4,3.793287771199935e-22).
nn('X0',5,4.213820776550114e-15).
nn('X0',6,2.7161646955035923e-13).
nn('X0',7,4.174621362300606e-15).
nn('X0',8,4.80026996681226e-15).
nn('X0',9,3.2756466077152183e-16).
nn('X1',0,2.2204615106602432e-06).
nn('X1',1,4.494424138101749e-05).
nn('X1',2,0.00039498970727436244).
nn('X1',3,3.255257979617454e-05).
nn('X1',4,1.2784033742718748e-06).
nn('X1',5,8.637762221042067e-05).
nn('X1',6,0.00031011790269985795).
nn('X1',7,0.0004717614792753011).
nn('X1',8,0.9985573887825012).
nn('X1',9,9.832474461290985e-05).
nn('X2',0,2.3827761985728557e-09).
nn('X2',1,1.0876910749857416e-07).
nn('X2',2,1.8467543938527342e-08).
nn('X2',3,2.2320566017697274e-07).
nn('X2',4,2.5658728191046976e-06).
nn('X2',5,0.9999425411224365).
nn('X2',6,4.752824054321536e-07).
nn('X2',7,4.32465697031148e-07).
nn('X2',8,1.1120650015072897e-05).
nn('X2',9,4.240589987603016e-05).
nn('X3',0,5.51503279012217e-15).
nn('X3',1,5.918688119905396e-14).
nn('X3',2,4.5902976989679704e-15).
nn('X3',3,7.17899465722932e-18).
nn('X3',4,1.2505416666978974e-20).
nn('X3',5,1.5132129761469965e-13).
nn('X3',6,2.9947993944991615e-24).
nn('X3',7,1.0).
nn('X3',8,8.908074194866405e-23).
nn('X3',9,2.703298349060468e-11).

a :- Pos=[f(['X0','X1','X2','X3'],20)], metaabd(Pos).
