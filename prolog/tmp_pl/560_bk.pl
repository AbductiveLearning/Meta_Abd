:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0410430655548308e-13).
nn('X0',1,2.4646398245922155e-16).
nn('X0',2,1.2358241140686784e-16).
nn('X0',3,1.3865245268186315e-17).
nn('X0',4,2.0690997614703694e-19).
nn('X0',5,4.715606702863917e-13).
nn('X0',6,3.6873478773712873e-25).
nn('X0',7,1.0).
nn('X0',8,6.599775425889229e-22).
nn('X0',9,3.020048655599794e-10).
nn('X1',0,1.3451636471373263e-09).
nn('X1',1,3.8012024106137687e-06).
nn('X1',2,0.00022350485960487276).
nn('X1',3,0.9997668266296387).
nn('X1',4,4.09093115286141e-13).
nn('X1',5,6.003919679642422e-06).
nn('X1',6,1.0217135026479446e-14).
nn('X1',7,2.9798782108514388e-08).
nn('X1',8,9.907567127820371e-11).
nn('X1',9,7.686133396411776e-10).
nn('X2',0,0.0008183749159798026).
nn('X2',1,7.291125569963697e-08).
nn('X2',2,0.0029148126486688852).
nn('X2',3,1.2360336087979817e-09).
nn('X2',4,5.164514732314274e-06).
nn('X2',5,0.0024000799749046564).
nn('X2',6,0.9931013584136963).
nn('X2',7,6.39650238554168e-07).
nn('X2',8,0.0007525882683694363).
nn('X2',9,6.933290023880545e-06).
nn('X3',0,3.185276001294035e-11).
nn('X3',1,3.812421559001855e-11).
nn('X3',2,5.018862062056151e-11).
nn('X3',3,1.9024789708055323e-06).
nn('X3',4,7.608497298861039e-07).
nn('X3',5,0.9784848093986511).
nn('X3',6,5.0808970745996884e-12).
nn('X3',7,0.00029719690792262554).
nn('X3',8,2.4514241303563722e-08).
nn('X3',9,0.021215302869677544).

a :- Pos=[f(['X0','X1'],10),f(['X2','X3'],11)], metaabd(Pos).