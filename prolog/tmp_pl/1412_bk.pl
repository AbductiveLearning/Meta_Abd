:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.9776552334936923e-09).
nn('X0',1,1.0).
nn('X0',2,1.0825271234971012e-11).
nn('X0',3,8.486054184065572e-22).
nn('X0',4,6.695044638040798e-14).
nn('X0',5,4.061172405311897e-12).
nn('X0',6,3.245663015693373e-13).
nn('X0',7,2.3339340876615822e-11).
nn('X0',8,7.846066190416584e-16).
nn('X0',9,6.752289157495317e-15).
nn('X1',0,7.972674513135087e-21).
nn('X1',1,2.0395298196005424e-22).
nn('X1',2,3.2200019011430286e-19).
nn('X1',3,8.99808352827769e-12).
nn('X1',4,1.9495971503857845e-09).
nn('X1',5,2.190993914075179e-11).
nn('X1',6,9.347122146959668e-24).
nn('X1',7,2.3690388843533583e-05).
nn('X1',8,1.3038784732900127e-14).
nn('X1',9,0.9999763369560242).
nn('X2',0,0.00016173161566257477).
nn('X2',1,4.8344681147227675e-08).
nn('X2',2,1.0784034287780742e-07).
nn('X2',3,3.4973295441886876e-06).
nn('X2',4,2.117117645639155e-07).
nn('X2',5,0.042565859854221344).
nn('X2',6,0.005981480237096548).
nn('X2',7,1.7841390217654407e-05).
nn('X2',8,0.9512630105018616).
nn('X2',9,6.281873083935352e-06).
nn('X3',0,2.3764719081498242e-09).
nn('X3',1,9.324116945208516e-07).
nn('X3',2,2.6925670681521297e-05).
nn('X3',3,0.9999663829803467).
nn('X3',4,4.263937293524922e-12).
nn('X3',5,5.119498382555321e-06).
nn('X3',6,1.6251933067541746e-14).
nn('X3',7,5.690409352610004e-07).
nn('X3',8,1.1852016590374603e-10).
nn('X3',9,7.551623326662593e-08).

a :- Pos=[f(['X0','X1','X2','X3'],21)], metaabd(Pos).
