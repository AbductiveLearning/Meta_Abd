:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.9998354315757751).
nn('X0',1,9.174518034171797e-09).
nn('X0',2,0.00016359436267521232).
nn('X0',3,1.3980294255588888e-08).
nn('X0',4,3.059024422658041e-10).
nn('X0',5,3.1731413940860875e-08).
nn('X0',6,8.147761718646507e-07).
nn('X0',7,1.6874746933126517e-08).
nn('X0',8,6.831555765529629e-08).
nn('X0',9,6.037100064304468e-08).
nn('X1',0,2.4590008848579537e-12).
nn('X1',1,1.5265212705006803e-10).
nn('X1',2,1.6816030665722792e-06).
nn('X1',3,4.7350650528952686e-17).
nn('X1',4,0.9999765753746033).
nn('X1',5,1.5148000784392934e-05).
nn('X1',6,7.559477666063685e-08).
nn('X1',7,3.1811686618254953e-09).
nn('X1',8,2.9889571720453922e-12).
nn('X1',9,6.59790975987562e-06).
nn('X2',0,3.34532259671505e-08).
nn('X2',1,8.92672105692327e-05).
nn('X2',2,0.00012338838132563978).
nn('X2',3,0.9967411756515503).
nn('X2',4,8.00919679022627e-07).
nn('X2',5,0.003042485099285841).
nn('X2',6,2.439146107313661e-10).
nn('X2',7,6.253304718484287e-07).
nn('X2',8,2.6943352438024704e-08).
nn('X2',9,2.250497573186294e-06).
nn('X3',0,1.2027442375384112e-09).
nn('X3',1,1.2431519702538196e-10).
nn('X3',2,1.0).
nn('X3',3,5.0203711026162917e-20).
nn('X3',4,7.214396349166048e-29).
nn('X3',5,1.3938924808475401e-22).
nn('X3',6,1.1994519237435377e-17).
nn('X3',7,1.804548128763947e-13).
nn('X3',8,6.960274206636009e-17).
nn('X3',9,7.661040982626857e-23).
nn('X4',0,1.473032980314759e-11).
nn('X4',1,1.5268371456045315e-08).
nn('X4',2,1.4365877177624498e-05).
nn('X4',3,6.083708688052079e-15).
nn('X4',4,0.9999459981918335).
nn('X4',5,2.7317573767504655e-05).
nn('X4',6,1.5093915450847817e-08).
nn('X4',7,1.220911372001865e-07).
nn('X4',8,1.0630647750975442e-10).
nn('X4',9,1.2172631613793783e-05).

a :- Pos=[f(['X0','X1','X2'],7),f(['X3','X4'],6)], metaabd(Pos).