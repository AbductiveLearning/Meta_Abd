:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.078228078516986e-08).
nn('X0',1,3.275977178418543e-07).
nn('X0',2,3.2609155375951104e-09).
nn('X0',3,2.149504929604973e-08).
nn('X0',4,4.046451351769065e-08).
nn('X0',5,0.9999873638153076).
nn('X0',6,6.400855312449494e-08).
nn('X0',7,1.07037828911416e-06).
nn('X0',8,1.0273362931911834e-05).
nn('X0',9,8.232208870140312e-07).
nn('X1',0,1.2694427731219093e-09).
nn('X1',1,4.1751859498617705e-06).
nn('X1',2,6.945119093870744e-05).
nn('X1',3,0.9997733235359192).
nn('X1',4,9.490524338628958e-11).
nn('X1',5,0.000152068751049228).
nn('X1',6,3.015030716410627e-13).
nn('X1',7,1.009770016935363e-06).
nn('X1',8,7.961979520132445e-09).
nn('X1',9,2.3078907673834692e-08).
nn('X2',0,4.9881872854484754e-08).
nn('X2',1,2.821572024913621e-06).
nn('X2',2,0.9999971389770508).
nn('X2',3,8.961754949243783e-12).
nn('X2',4,4.598999094727185e-19).
nn('X2',5,1.5956176267414857e-14).
nn('X2',6,2.4238196085629715e-13).
nn('X2',7,9.74362279748675e-08).
nn('X2',8,2.2639091989962168e-11).
nn('X2',9,2.085265689505747e-14).
nn('X3',0,1.1156748674868641e-14).
nn('X3',1,1.758661135285982e-14).
nn('X3',2,1.0684947257489849e-13).
nn('X3',3,3.199219520411134e-07).
nn('X3',4,4.3953414774478006e-07).
nn('X3',5,3.243356516691165e-08).
nn('X3',6,5.442692258899272e-17).
nn('X3',7,0.0006825271993875504).
nn('X3',8,6.154873277708361e-10).
nn('X3',9,0.9993166923522949).
nn('X4',0,0.009669965133070946).
nn('X4',1,7.638393435627222e-05).
nn('X4',2,2.349662509004702e-06).
nn('X4',3,2.0635810869862325e-05).
nn('X4',4,4.770311716129072e-05).
nn('X4',5,0.9353221654891968).
nn('X4',6,0.022662336006760597).
nn('X4',7,0.001053046784363687).
nn('X4',8,0.030589791014790535).
nn('X4',9,0.0005556291434913874).
nn('X5',0,7.286354759804823e-16).
nn('X5',1,1.3495437135423866e-15).
nn('X5',2,6.466975577268386e-08).
nn('X5',3,4.2482439554167574e-24).
nn('X5',4,0.9999997615814209).
nn('X5',5,8.550780705718353e-08).
nn('X5',6,2.126301801297359e-09).
nn('X5',7,6.042196365636798e-14).
nn('X5',8,8.514894862020063e-19).
nn('X5',9,6.3757399360042655e-09).
nn('X6',0,3.3972309410046364e-09).
nn('X6',1,9.802762591155261e-17).
nn('X6',2,1.5530467190050246e-10).
nn('X6',3,1.7261616654983946e-19).
nn('X6',4,1.5440544842704185e-08).
nn('X6',5,1.1745337360480335e-06).
nn('X6',6,0.9999988079071045).
nn('X6',7,6.377201474990076e-16).
nn('X6',8,3.4485757213200893e-14).
nn('X6',9,7.640637936696959e-16).

a :- Pos=[f(['X0','X1','X2'],10),f(['X3','X4','X5','X6'],24)], metaabd(Pos).
