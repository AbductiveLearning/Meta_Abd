:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,2.0916068289500544e-18).
nn('X0',2,6.7031205889731e-14).
nn('X0',3,7.884996437820021e-20).
nn('X0',4,1.8573156117120175e-24).
nn('X0',5,3.4262989580952086e-13).
nn('X0',6,5.9429610348550946e-15).
nn('X0',7,2.3368147694657893e-12).
nn('X0',8,2.5543077580925153e-17).
nn('X0',9,1.4601832232871923e-17).
nn('X1',0,0.9999966621398926).
nn('X1',1,1.6227996594078697e-13).
nn('X1',2,1.4641689460859197e-07).
nn('X1',3,4.940731157532241e-11).
nn('X1',4,2.106919510819899e-13).
nn('X1',5,1.239296693711367e-06).
nn('X1',6,2.015037125602248e-06).
nn('X1',7,5.276645786977952e-10).
nn('X1',8,4.186868807209976e-08).
nn('X1',9,6.491400195152153e-10).
nn('X2',0,3.0758013025433684e-08).
nn('X2',1,6.586643053196894e-08).
nn('X2',2,7.969987336764461e-09).
nn('X2',3,4.438430778463953e-08).
nn('X2',4,4.9735852769572375e-08).
nn('X2',5,0.9999990463256836).
nn('X2',6,4.2777759290402173e-07).
nn('X2',7,5.287368320949781e-09).
nn('X2',8,3.672422721479052e-08).
nn('X2',9,2.497173454685253e-07).
nn('X3',0,1.6769503537530928e-12).
nn('X3',1,1.1198241092009661e-13).
nn('X3',2,2.4634993697758567e-11).
nn('X3',3,4.672230886626494e-07).
nn('X3',4,7.9146684583975e-06).
nn('X3',5,5.8480779330238875e-08).
nn('X3',6,4.115989911098253e-14).
nn('X3',7,0.0012641212670132518).
nn('X3',8,2.2808899657889015e-08).
nn('X3',9,0.9987274408340454).
nn('X4',0,3.220571365147862e-09).
nn('X4',1,7.074603802180638e-17).
nn('X4',2,4.254745722409581e-10).
nn('X4',3,4.531655736119829e-19).
nn('X4',4,4.956920065524173e-07).
nn('X4',5,7.936235419947479e-07).
nn('X4',6,0.9999986886978149).
nn('X4',7,2.1990079841684856e-14).
nn('X4',8,2.0265901925901365e-14).
nn('X4',9,2.6772787003852272e-14).
nn('X5',0,1.0490971806553784e-11).
nn('X5',1,2.205116311770894e-09).
nn('X5',2,7.010723379607953e-08).
nn('X5',3,8.144249719066465e-09).
nn('X5',4,7.386904976591779e-10).
nn('X5',5,4.507096207362338e-08).
nn('X5',6,3.307613072323079e-10).
nn('X5',7,3.485208435449749e-05).
nn('X5',8,0.9999542832374573).
nn('X5',9,1.07654059320339e-05).

a :- Pos=[f(['X0','X1','X2','X3'],14),f(['X4','X5'],14)], metaabd(Pos).