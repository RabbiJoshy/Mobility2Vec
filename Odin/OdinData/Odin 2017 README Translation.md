# Input data
This folder contains inputdata used through the project.

## fused_dataset_{pickled.bz2, plain.csv}
This contains the ODiN data fused with KNMI rain, OTP alternatives and travel cost data in either the pandas dataframe format (`_pickled.bz2`) or as a plain CSV (`_plain.csv`).
If using pandas it's recommended to read the `bz2` file directly with [`pandas.read_pickle()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html).  

This file is created by the script [`fuse_odin_otp_knmi.py`](https://ci.tno.nl/gitlab/SustainableUrbanMobilityAndSafety/mode-choice-ai-dcm/-/blob/master/odin/fuse_odin_otp_knmi.py).

Column description (sorry, in Dutch as I had to type it in Dutch for a report).

- trip_id: unieke verplaatsing ID uit ODiN
- sted_{o, d}: degree of urbanization origin (o) or destination (d) postal code;
  - 1: Very highly urban (area address density of 2 500 or more)
  - 2: Highly urban (environmental address density of 1,500 to 2,500)
  - 3: Moderately urban (area address density from 1,000 to 1,500)
  - 4: Little urban (area address density from 500 to 1,000)
  - 5: Non-urban (area address density less than 500)
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
- pur_{home, work, busn, other}: the purpose of the trip is (1) or not (0) 
going home, work, business or other.
- {departure, arrival}_rain: neerslag in mm/15min bij de herkomst (departure) of bestemming (arrival).

- choice: de gekozen modaliteit volgens ODiN;
  - 1: Auto bestuurder
  - 2. Auto passagier
  - 3: OV (bus, tram, metro of trein)
  - 4: Fiets
  - 5: Lopen

- dist_{car, carp, transit, cycle, walk}: distance of the trip or the alternative by car (car), as a passenger (carp), by public transport (transit), bicycle (cycle) or on foot (walk) ) in meters.
- t_{car, carp, transit, cycle, walk}: travel time of the journey or the alternative by car (car), as a passenger (carp), by public transport (transit), bicycle (cycle) or on foot (walk) ) in seconds.
- c_{car, carp, transit, cycle, walk}: costs of the journey or the alternative by car (car), as a passenger (carp), by public transport (transit), bicycle (cycle) or on foot (walk) ) in euros.
- {vc, pc}_car: distance dependent (vc) and parking (pc) costs for the car driver.
- av_{car, carp, transit, cycle, walk}: whether (1) or not (0) the modality is available for a journey or alternative.

- activity duration: activity duration at the destination in minutes from ODiN.
- trip duration_sec: actual trip duration in seconds from ODiN.
- afstv_m: actual displacement distance in meters from ODiN.
- aankpc: numeric part of the destination zip code from ODiN
- 
- creation_datetime: de datum en tijd waarop het bestand is gemaakt, alleen de eerste rij heeft een waarde.

