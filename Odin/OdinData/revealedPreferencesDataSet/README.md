# Input data
This folder contains inputdata used through the project.

## fused_dataset_{pickled.bz2, plain.csv}
This contains the ODiN data fused with KNMI rain, OTP alternatives and travel cost data in either the pandas dataframe format (`_pickled.bz2`) or as a plain CSV (`_plain.csv`).
If using pandas it's recommended to read the `bz2` file directly with [`pandas.read_pickle()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html).  

This file is created by the script [`fuse_odin_otp_knmi.py`](https://ci.tno.nl/gitlab/SustainableUrbanMobilityAndSafety/mode-choice-ai-dcm/-/blob/master/odin/fuse_odin_otp_knmi.py).

Column description (sorry, in Dutch as I had to type it in Dutch for a report).

- trip_id: unieke verplaatsing ID uit ODiN
- sted_{o, d}: stedelijkheidsgraad herkomst (o) of bestemming (d) postcode;
  - 1: Zeer sterk stedelijk (omgevingsadressendichtheid van 2 500 of meer)
  - 2: Sterk stedelijk (omgevingsadressendichtheid van 1 500 tot 2 500)
  - 3: Matig stedelijk (omgevingsadressendichtheid van 1 000 tot 1 500)
  - 4: Weinig stedelijk (omgevingsadressendichtheid van 500 tot 1 000)
  - 5: Niet-stedelijk (omgevingsadressendichtheid van minder dan 500)
- ovstkaart: geeft aan of een reiziger een week (1), weekend (2) of geen OV-studentenkaart heeft (0).
- weekday: geeft de dag van de week aan; 
  - 1: zondag
  - 2: maandag
  - ...
  - 7: zaterdag
- d_hhchildren: wel (1) of geen (0) kinderen in het huishouden. 
- d_high_educ: de reiziger heeft wel (1) of niet (0) een HBO opleiding of hoger afgerond.
- gender: geslacht van de reiziger; 
  - 0: vrouw
  - 1: man
- age: leeftijdsgroep van de reiziger;
  - 1: 6 t/m 17 jaar
  - 2: 18 t/m 54 jaar
  - 3: 55+
- driving_license: de reiziger heeft wel (1) of geen (0)  autorijbewijs.
- car_ownership: het huishouden heeft wel (1) of geen (0) auto.
- main_car_user: de reiziger heeft wel (1) of geen (0) auto op zijn/haar naam.
- hh_highinc10: het huishouden heeft wel (1) of geen (0) top 10% inkomen.
- hh_lowinc10: het huishouden heeft wel (1) of geen (0) onderste 10% inkomen.
- hh_highinc20: het huishouden heeft wel (1) of geen (0) top 20% inkomen.
- hh_lowinc20: het huishouden heeft wel (1) of geen (0) onderste 20% inkomen.
- pur_{home, work, busn, other}: het doel van de reis is wel (1) of niet (0) naar huis gaan, werken, zakelijk of overig.
- {departure, arrival}_rain: neerslag in mm/15min bij de herkomst (departure) of bestemming (arrival).
- choice: de gekozen modaliteit volgens ODiN;
  - 1: Auto bestuurder
  - 2. Auto passagier
  - 3: OV (bus, tram, metro of trein)
  - 4: Fiets
  - 5: Lopen
- dist_{car, carp, transit, cycle, walk}: afstand van de verplaatsing of het alternatief met de auto (car), als passagier (carp), met het OV (transit), fiets (cycle) of te voet (walk) in meter.
- t_{car, carp, transit, cycle, walk}: reistijd van de verplaatsing of het alternatief met de auto (car), als passagier (carp), met het OV (transit), fiets (cycle) of te voet (walk) in seconden.
- c_{car, carp, transit, cycle, walk}: kosten van de verplaatsing of het alternatief met de auto (car), als passagier (carp), met het OV (transit), fiets (cycle) of te voet (walk) in euro.
- {vc, pc}_car: afstandsafhankelijke- (vc) en parkeer- (pc) kosten voor de autobestuurder.
- av_{car, carp, transit, cycle, walk}: wel (1) of niet (0) beschikbaar zijn van de modaliteit voor een verplaatsing of alternatief.
- actduur: activiteitduur op de bestemming in minuten uit ODiN.
- reisduur_sec: daadwerkelijke reisduur in seconden uit ODiN.
- afstv_m: daadwerkelijke verplaatsingsafstand in meter uit ODiN.
- aankpc: numerieke deel van de bestemmingspostcode uit ODiN
- creation_datetime: de datum en tijd waarop het bestand is gemaakt, alleen de eerste rij heeft een waarde.

