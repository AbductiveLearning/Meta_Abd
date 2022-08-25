:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.0005690255202353001).
nn('X0',1,2.047298494289862e-06).
nn('X0',2,0.9993273019790649).
nn('X0',3,7.802818657864918e-08).
nn('X0',4,2.172987251469749e-06).
nn('X0',5,2.8913760274917877e-07).
nn('X0',6,7.565807754872367e-05).
nn('X0',7,1.8582230154606805e-07).
nn('X0',8,2.2978138076723553e-05).
nn('X0',9,2.446907956255018e-07).
nn('X1',0,4.4390332846978766e-15).
nn('X1',1,1.4350450081718202e-12).
nn('X1',2,1.6858186721591117e-11).
nn('X1',3,6.008166906212864e-07).
nn('X1',4,0.0008902138797566295).
nn('X1',5,5.384723408496939e-06).
nn('X1',6,2.6166316495815876e-14).
nn('X1',7,0.0009240685030817986).
nn('X1',8,7.486039343973516e-09).
nn('X1',9,0.9981797933578491).
nn('X2',0,0.0014281023759394884).
nn('X2',1,0.0006532046827487648).
nn('X2',2,0.008487209677696228).
nn('X2',3,0.003671378130093217).
nn('X2',4,0.00494199525564909).
nn('X2',5,0.9802797436714172).
nn('X2',6,3.87869804399088e-05).
nn('X2',7,0.00022612803149968386).
nn('X2',8,2.930468508566264e-05).
nn('X2',9,0.0002441634424030781).
nn('X3',0,1.0).
nn('X3',1,4.8737571027130546e-17).
nn('X3',2,6.855497419744339e-11).
nn('X3',3,4.1460308553900845e-19).
nn('X3',4,9.767994907879708e-20).
nn('X3',5,1.577729137928422e-14).
nn('X3',6,3.030083508322873e-13).
nn('X3',7,7.073178414501724e-12).
nn('X3',8,1.3418640893261054e-15).
nn('X3',9,4.3184761864580035e-14).
nn('X4',0,2.150066791273275e-08).
nn('X4',1,1.3044419060539991e-12).
nn('X4',2,8.044600402490332e-11).
nn('X4',3,8.349144401016695e-13).
nn('X4',4,5.677974718240467e-16).
nn('X4',5,1.9614010415835992e-10).
nn('X4',6,7.824803409373332e-19).
nn('X4',7,0.9999998807907104).
nn('X4',8,1.929934288891059e-15).
nn('X4',9,1.6823160819967597e-07).
nn('X5',0,3.0123139538318355e-08).
nn('X5',1,0.9999998807907104).
nn('X5',2,6.188519918737256e-09).
nn('X5',3,1.9246966705878794e-18).
nn('X5',4,1.3885507327282554e-11).
nn('X5',5,3.075689225875977e-12).
nn('X5',6,6.077892637963711e-13).
nn('X5',7,6.593091228523917e-08).
nn('X5',8,1.5442075786478715e-12).
nn('X5',9,2.0038413203055105e-12).

a :- Pos=[f(['X0','X1','X2'],16),f(['X3','X4','X5'],8)], metaabd(Pos).
