:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,9.846644688087322e-18).
nn('X0',2,6.197410917785362e-10).
nn('X0',3,4.583868927347825e-18).
nn('X0',4,7.767372000949834e-20).
nn('X0',5,2.590456976258714e-14).
nn('X0',6,1.0589757152776325e-12).
nn('X0',7,3.584765080263952e-12).
nn('X0',8,1.336577875286904e-13).
nn('X0',9,1.652972768698916e-14).
nn('X1',0,6.77187870223861e-08).
nn('X1',1,5.349289618168185e-15).
nn('X1',2,2.8465765300467183e-09).
nn('X1',3,2.9383533235582276e-17).
nn('X1',4,1.016180064539185e-07).
nn('X1',5,2.6340933345636586e-06).
nn('X1',6,0.9999973177909851).
nn('X1',7,1.8582084113615005e-13).
nn('X1',8,4.527688553940257e-12).
nn('X1',9,2.7646257543456274e-14).
nn('X2',0,8.206913548747252e-07).
nn('X2',1,0.9999985694885254).
nn('X2',2,2.8735081514241756e-07).
nn('X2',3,2.206256282267366e-13).
nn('X2',4,4.115129126347483e-09).
nn('X2',5,6.49184102030631e-08).
nn('X2',6,8.379888782883427e-09).
nn('X2',7,2.896178159517149e-07).
nn('X2',8,3.6890031029912507e-09).
nn('X2',9,4.271555109625069e-09).
nn('X3',0,3.4489221434341744e-06).
nn('X3',1,0.009889491833746433).
nn('X3',2,0.00031588933779858053).
nn('X3',3,0.032452430576086044).
nn('X3',4,0.09892655164003372).
nn('X3',5,0.028132988139986992).
nn('X3',6,1.372771691876551e-07).
nn('X3',7,0.004017469007521868).
nn('X3',8,0.0010229827603325248).
nn('X3',9,0.8252387046813965).
nn('X4',0,9.089847472810997e-13).
nn('X4',1,5.237087822326618e-15).
nn('X4',2,6.358687990211387e-13).
nn('X4',3,1.3143602473064675e-07).
nn('X4',4,1.2179633301911963e-07).
nn('X4',5,1.007579530920566e-08).
nn('X4',6,8.086300616125665e-16).
nn('X4',7,0.0003897486603818834).
nn('X4',8,1.7237390403934683e-09).
nn('X4',9,0.9996100068092346).
nn('X5',0,2.9800885315012238e-08).
nn('X5',1,4.254725809005322e-06).
nn('X5',2,2.6224444809486158e-05).
nn('X5',3,2.910574721681769e-06).
nn('X5',4,2.3313360486554302e-07).
nn('X5',5,2.5081311832764186e-05).
nn('X5',6,2.0369814592413604e-05).
nn('X5',7,0.0001845842634793371).
nn('X5',8,0.9996926784515381).
nn('X5',9,4.370222086436115e-05).
nn('X6',0,0.0014309701509773731).
nn('X6',1,2.795816271827789e-07).
nn('X6',2,9.986039772869049e-10).
nn('X6',3,2.6574601894036043e-12).
nn('X6',4,2.2535675547885603e-09).
nn('X6',5,0.9980806708335876).
nn('X6',6,0.0004611681215465069).
nn('X6',7,7.894675945863128e-06).
nn('X6',8,1.9105746105196886e-05).
nn('X6',9,2.8505264815237297e-09).

a :- Pos=[f(['X0','X1','X2'],7),f(['X3','X4','X5','X6'],31)], metaabd(Pos).