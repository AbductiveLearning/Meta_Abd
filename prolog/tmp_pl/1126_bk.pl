:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,3.733974240915927e-10).
nn('X0',1,1.7331759291638882e-11).
nn('X0',2,7.826155723478223e-09).
nn('X0',3,5.029020258007222e-07).
nn('X0',4,0.0009572759154252708).
nn('X0',5,4.6321924855874386e-06).
nn('X0',6,3.1704214253913676e-10).
nn('X0',7,0.0037705013528466225).
nn('X0',8,5.033352863392793e-07).
nn('X0',9,0.9952664971351624).
nn('X1',0,4.0846985148965746e-18).
nn('X1',1,5.740950106196255e-20).
nn('X1',2,1.5220160328667152e-16).
nn('X1',3,7.921677896982615e-11).
nn('X1',4,2.0906362863115646e-08).
nn('X1',5,1.4480293353269502e-10).
nn('X1',6,3.506172239574366e-21).
nn('X1',7,4.66314195364248e-05).
nn('X1',8,1.8711069613196532e-14).
nn('X1',9,0.9999534487724304).
nn('X2',0,1.0).
nn('X2',1,4.4769374139674065e-23).
nn('X2',2,3.81728187877654e-15).
nn('X2',3,1.4480040475423771e-22).
nn('X2',4,1.5320711581894218e-30).
nn('X2',5,1.189736919059674e-16).
nn('X2',6,1.2512361432417664e-16).
nn('X2',7,2.4326015974557096e-16).
nn('X2',8,9.741691598376287e-19).
nn('X2',9,5.855271879067982e-21).
nn('X3',0,9.334317496723088e-08).
nn('X3',1,8.866396819939837e-05).
nn('X3',2,0.0008994305972009897).
nn('X3',3,0.998813271522522).
nn('X3',4,3.653164715178292e-10).
nn('X3',5,0.00016900437185540795).
nn('X3',6,1.0706833336593391e-10).
nn('X3',7,2.8963686418137513e-05).
nn('X3',8,5.563279046327807e-07).
nn('X3',9,6.06545853543139e-08).
nn('X4',0,1.2330532150883755e-08).
nn('X4',1,1.0).
nn('X4',2,4.014829357856797e-09).
nn('X4',3,8.702698982269869e-19).
nn('X4',4,2.1715646641995434e-11).
nn('X4',5,1.506774566228941e-11).
nn('X4',6,2.2203402311182785e-12).
nn('X4',7,9.921892640818442e-09).
nn('X4',8,2.087715200368989e-12).
nn('X4',9,3.790500899270022e-12).

a :- Pos=[f(['X0','X1'],18),f(['X2','X3','X4'],4)], metaabd(Pos).
