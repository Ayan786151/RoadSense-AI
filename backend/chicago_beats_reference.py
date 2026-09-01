"""
================================================================================
ROADSENSE AI — AUTHENTIC CHICAGO POLICE BEAT TO NEIGHBORHOOD DIRECTORY
FILE: backend/chicago_beats_reference.py
================================================================================

Comprehensive mapping for all 275 Chicago Police Department (CPD) beats
to authentic Chicago Community Areas, famous neighborhoods, landmarks,
and arterial corridors.
================================================================================
"""

from typing import Dict, Any

# Primary CPD Districts and their community areas
CPD_DISTRICT_PROFILES = {
    1: {"district": "01st District - Central", "neighborhood": "Downtown Loop & South Loop", "type": "High-Density Commercial & Mixed Transit"},
    2: {"district": "02nd District - Wentworth", "neighborhood": "Bronzeville, Hyde Park & Kenwood", "type": "Historic Urban & University Boulevard"},
    3: {"district": "03rd District - Grand Crossing", "neighborhood": "Woodlawn, South Shore & Jackson Park", "type": "Lakefront Arterial & Mixed Residential"},
    4: {"district": "04th District - South Chicago", "neighborhood": "South Chicago, Avalon Park & Hegewisch", "type": "Industrial, Freight & Lakefront Corridor"},
    5: {"district": "05th District - Calumet", "neighborhood": "Roseland, Pullman & West Pullman", "type": "Historic Industrial & Transit Corridor"},
    6: {"district": "06th District - Gresham", "neighborhood": "Chatham, Auburn Gresham & Park Manor", "type": "Major Arterial Crossroad Commercial"},
    7: {"district": "07th District - Englewood", "neighborhood": "Englewood & West Englewood", "type": "High-Volume Grid Crossroad"},
    8: {"district": "08th District - Chicago Lawn", "neighborhood": "Midway Airport, Clearing & West Lawn", "type": "Airport Logistics & Dense Commercial"},
    9: {"district": "09th District - Deering", "neighborhood": "Bridgeport, McKinley Park & Back of the Yards", "type": "Industrial, Rail & Arts Corridor"},
    10: {"district": "10th District - Ogden", "neighborhood": "Little Village & North Lawndale", "type": "Dense Commercial Corridor (26th St)"},
    11: {"district": "11th District - Harrison", "neighborhood": "East & West Garfield Park", "type": "Arterial Expressway Merge (I-290)"},
    12: {"district": "12th District - Near West", "neighborhood": "West Loop, Fulton Market, UIC & Pilsen", "type": "Dining, Medical District & Arts Hub"},
    14: {"district": "14th District - Shakespeare", "neighborhood": "Wicker Park, Bucktown & Logan Square", "type": "Six-Way Complex Junctions & Boulevards"},
    15: {"district": "15th District - Austin", "neighborhood": "Austin & Galewood", "type": "High-Volume Commercial Arterials"},
    16: {"district": "16th District - Jefferson Park", "neighborhood": "O'Hare Airport, Jefferson Park & Norwood Park", "type": "Airport Terminal Core & Expressway Hub"},
    17: {"district": "17th District - Albany Park", "neighborhood": "Albany Park, Irving Park & Avondale", "type": "Global Transit Corridor & River Junction"},
    18: {"district": "18th District - Near North", "neighborhood": "River North, Magnificent Mile, Gold Coast & Streeterville", "type": "Luxury Commercial & Waterfront"},
    19: {"district": "19th District - Town Hall", "neighborhood": "Lincoln Park, Lakeview, Wrigleyville & Boystown", "type": "High Pedestrian Mixed & Sports Corridor"},
    20: {"district": "20th District - Lincoln", "neighborhood": "Andersonville, Edgewater & Lincoln Square", "type": "Cultural Commercial & Lakefront Transit"},
    22: {"district": "22nd District - Morgan Park", "neighborhood": "Beverly, Morgan Park & Mount Greenwood", "type": "Historic Residential & Ridge Arterial"},
    24: {"district": "24th District - Rogers Park", "neighborhood": "Rogers Park, West Ridge & Loyola Campus", "type": "University Lakefront & Multi-Ethnic Corridor"},
    25: {"district": "25th District - Grand Central", "neighborhood": "Belmont Cragin, Hermosa & Kelvyn Park", "type": "High-Density Commercial & Freight Rail"},
    31: {"district": "31st District - Special Operations", "neighborhood": "Citywide Expressway & Traffic Command", "type": "Municipal Transit Backbone"},
    61: {"district": "06th District - Gresham", "neighborhood": "Chatham / 79th St Commercial Gateway", "type": "Major Commercial Intersection"}
}

# Detailed Corridor Overrides for Distinct Police Beats
EXACT_CHICAGO_BEAT_DETAILS = {
    # District 1 - Central (The Loop & South Loop)
    "111": "Downtown Loop - Wacker Dr / Michigan Ave / Riverwalk",
    "112": "Downtown Loop - Madison St / LaSalle St Financial Corridor",
    "113": "Downtown Loop - Adams St / State St Shopping Corridor",
    "114": "Downtown Loop - Congress Pkwy / Van Buren Transit Gateway",
    "121": "Printers Row - Harrison St / Dearborn St Historic District",
    "122": "South Loop - Roosevelt Rd / Michigan Ave Retail Center",
    "123": "South Loop - Museum Campus / Lake Shore Dr / Soldier Field",
    "124": "South Loop - Cermak Rd / Motor Row Historic District",
    "131": "Chinatown - Cermak Rd / Wentworth Ave Gate",
    "132": "Prairie District - 18th St / Indiana Ave Historic Mansions",
    "133": "McCormick Place - Cermak Rd / Martin Luther King Dr",

    # District 2 - Wentworth (Bronzeville & Hyde Park)
    "211": "Bronzeville - 31st St / King Dr Parkway",
    "212": "Bronzeville - 35th St / Cottage Grove Arts Corridor",
    "213": "Bronzeville - 39th St / Pershing Rd Gateway",
    "214": "Oakland - Lake Park Ave / 43rd St",
    "215": "Kenwood - 47th St / Drexel Blvd Historic Boulevard",
    "221": "Grand Boulevard - 43rd St / State St Green Line",
    "222": "Grand Boulevard - 47th St / King Dr Commercial",
    "223": "Washington Park - 51st St / King Dr",
    "224": "Washington Park - Garfield Blvd / South Park Gateway",
    "225": "Washington Park - 59th St / Cottage Grove Green Line",
    "231": "Hyde Park - 53rd St Downtown Commercial Center",
    "232": "Hyde Park - 55th St / University of Chicago North",
    "233": "Hyde Park - University of Chicago Main Quad & Hospitals",
    "234": "Hyde Park - Museum of Science & Industry / 57th St Beach",
    "235": "Hyde Park - 60th St / Midway Plaisance Boulevard",

    # District 3 - Grand Crossing (Woodlawn & South Shore)
    "311": "Woodlawn - 63rd St / Cottage Grove Green Line Terminal",
    "312": "Woodlawn - 67th St / King Dr",
    "313": "Woodlawn - Jackson Park / Stony Island Ave",
    "314": "Woodlawn - 63rd St Beach & Harbor / Lake Shore Dr",
    "321": "Greater Grand Crossing - 71st St / State St Retail",
    "322": "Greater Grand Crossing - 75th St / King Dr Corridor",
    "323": "Greater Grand Crossing - 71st St / Cottage Grove Hub",
    "324": "Greater Grand Crossing - 75th St / Stony Island Crossroads",
    "331": "South Shore - 67th St / South Shore Cultural Center",
    "332": "South Shore - 71st St Commercial Corridor",
    "333": "South Shore - 75th St / Jeffery Blvd Transit Corridor",
    "334": "South Shore - 79th St / South Shore Dr Lakefront",

    # District 4 - South Chicago
    "411": "Avalon Park - 79th St / Stony Island Ave Hub",
    "412": "Avalon Park - 83rd St / South Chicago Ave Diagonal",
    "413": "South Chicago - Commercial Ave Historic Shopping Strip",
    "414": "South Chicago - 87th St / Chicago Skyway Approach",
    "421": "Calumet Heights - 87th St / Jeffery Blvd",
    "422": "Calumet Heights - 91st St / Stony Island Ave",
    "423": "South Deering - 95th St / Torrence Ave Industrial Hub",
    "424": "South Deering - 100th St / Commercial Ave",
    "431": "East Side - 106th St / Indianapolis Ave State Line",
    "432": "East Side - 112th St / Ewing Ave Calumet River",
    "433": "Hegewisch - 130th St / Brainard Ave Freight Rail",
    "434": "Hegewisch - Torrence Ave Ford Assembly Campus",

    # District 5 - Calumet
    "511": "Roseland - 95th St Red Line Transit Terminal",
    "512": "Roseland - 103rd St / Michigan Ave Commercial Strip",
    "513": "Roseland - 111th St / State St",
    "521": "Pullman - National Historic Park / 111th St Arcade",
    "522": "Pullman - 115th St / Cottage Grove Ave",
    "523": "West Pullman - 119th St / Halsted St Commercial",
    "524": "West Pullman - 123rd St / Michigan Ave",
    "531": "Riverdale - 130th St / Altgeld Gardens",
    "532": "Riverdale - 134th St / Beaubien Woods Marina",
    "533": "Riverdale - 138th St / Calumet River Industrial Gateway",

    # District 6 - Gresham
    "611": "Chatham - 79th St / State St Retail Corridor",
    "612": "Chatham - 83rd St / Cottage Grove Ave",
    "613": "Chatham - 87th St / King Dr Commercial Center",
    "621": "Park Manor - 71st St / South Park Corridor",
    "622": "Auburn Gresham - 76th St / Vincennes Ave",
    "623": "Auburn Gresham - 79th St / Halsted St Major Crossroad",
    "624": "Auburn Gresham - 83rd St / Racine Ave",
    "631": "Auburn Gresham - 87th St / Ashland Ave Commercial Hub",
    "632": "Auburn Gresham - 79th St / Western Ave Transit Corridor",
    "633": "Ashburn East - 83rd St / Kedzie Ave",
    "634": "Ashburn East - 87th St / Columbus Ave Diagonal",

    # District 7 - Englewood
    "711": "Englewood - 59th St / Halsted St Corridor",
    "712": "Englewood - 63rd St / Green Line Transit Center",
    "713": "Englewood - 63rd St / Halsted St Commercial Square",
    "714": "Englewood - 67th St / Racine Ave",
    "715": "Englewood - 71st St / Halsted St Retail",
    "721": "West Englewood - 59th St / Ashland Ave",
    "722": "West Englewood - 63rd St / Damen Ave",
    "723": "West Englewood - 67th St / Western Ave Corridor",
    "724": "West Englewood - 69th St / Ashland Ave Crossroad",
    "725": "West Englewood - 71st St / Damen Ave",
    "731": "Hamilton Park - 72nd St / Normal Ave Cultural Center",
    "732": "Hamilton Park - 74th St / Parnell Ave",

    # District 8 - Chicago Lawn
    "811": "Gage Park - 51st St / Kedzie Ave Commercial",
    "812": "Gage Park - 55th St / Western Ave Hub",
    "813": "Gage Park - 59th St / California Ave",
    "814": "Chicago Lawn - 63rd St / Western Ave Marquette Park",
    "815": "Chicago Lawn - 67th St / Kedzie Ave",
    "821": "West Lawn - 63rd St / Pulaski Rd Commercial Center",
    "822": "West Lawn - 67th St / Cicero Ave",
    "823": "Clearing - Midway Airport 55th St Cargo Corridor",
    "824": "Clearing - 63rd St / Central Ave Midway Terminal",
    "825": "Garfield Ridge - Archer Ave / 55th St Orange Line",
    "831": "Garfield Ridge - Archer Ave / Harlem Ave Gateway",
    "832": "Clearing - 65th St / Cicero Ave Industrial Strip",
    "833": "Ashburn - 79th St / Cicero Ave Ford City Mall",
    "834": "Ashburn - 83rd St / Pulaski Rd",
    "835": "Ashburn - 87th St / Kedzie Ave St. Xavier Campus",

    # District 9 - Deering
    "911": "Bridgeport - 31st St / Halsted St Arts District",
    "912": "Bridgeport - 35th St / Morgan St Historic Strip",
    "913": "Bridgeport - Sox-35th Guaranteed Rate Field Stadium Hub",
    "914": "McKinley Park - 35th St / Archer Ave Orange Line Hub",
    "915": "McKinley Park - 39th St / Western Blvd Corridor",
    "921": "Brighton Park - Archer Ave / Kedzie Ave Orange Line",
    "922": "Brighton Park - 43rd St / California Ave",
    "923": "Brighton Park - 47th St / Kedzie Ave Commercial",
    "924": "Back of the Yards - 43rd St / Ashland Ave Retail",
    "925": "Back of the Yards - 47th St / Halsted St Historic",
    "931": "Back of the Yards - 51st St / Damen Ave",
    "932": "Back of the Yards - 51st St / Ashland Ave Industrial",
    "933": "New City - 49th St / Racine Ave",
    "934": "Canaryville - 43rd St / Union Ave",
    "935": "Canaryville - 47th St / Normal Ave",

    # District 10 - Ogden
    "1011": "North Lawndale - Roosevelt Rd / Pulaski Rd Crossroads",
    "1012": "North Lawndale - 16th St / Kostner Ave",
    "1013": "North Lawndale - Cermak Rd / Pulaski Rd Transit Hub",
    "1014": "North Lawndale - Ogden Ave / Douglas Park Boulevard",
    "1021": "North Lawndale - Roosevelt Rd / Kedzie Ave",
    "1022": "North Lawndale - 16th St / Central Park Ave",
    "1023": "North Lawndale - Cermak Rd / Kedzie Ave Pink Line",
    "1024": "North Lawndale - 21st St / California Ave",
    "1031": "Little Village - 26th St / Pulaski Rd Mexican Commercial Hub",
    "1032": "Little Village - 26th St / Kedzie Ave Historic Arch",
    "1033": "Little Village - 31st St / Kedzie Ave Industrial",
    "1034": "South Lawndale - 31st St / Kostner Ave Corridor",

    # District 11 - Harrison
    "1111": "West Garfield Park - Madison St / Pulaski Rd Crossroads",
    "1112": "West Garfield Park - Lake St / Pulaski Rd Green Line",
    "1113": "West Garfield Park - Chicago Ave / Kostner Ave",
    "1114": "West Garfield Park - Jackson Blvd / Independence Blvd",
    "1115": "West Garfield Park - Harrison St / Cicero Ave I-290",
    "1121": "East Garfield Park - Conservatory / Central Park Green Line",
    "1122": "East Garfield Park - Madison St / Kedzie Ave",
    "1123": "East Garfield Park - Franklin Blvd / Sacramento Blvd",
    "1124": "East Garfield Park - Harrison St / Kedzie Ave Blue Line",
    "1125": "East Garfield Park - Fifth Ave / California Ave",
    "1131": "Humboldt Park South - Chicago Ave / Kedzie Ave",
    "1132": "Humboldt Park South - Augusta Blvd / California Ave",
    "1133": "Humboldt Park South - Grand Ave / Sacramento Blvd",
    "1134": "East Garfield - Warren Blvd / Western Ave",
    "1135": "East Garfield - Eisenhower Expressway / Oakley Blvd",

    # District 12 - Near West
    "1211": "West Loop - Fulton Market / Randolph St Restaurant Row",
    "1212": "West Loop - Madison St / United Center Sports Arena",
    "1213": "West Loop - Halsted St / Adams St Historic Greektown",
    "1214": "West Loop - Van Buren St / Racine Ave Blue Line",
    "1215": "Illinois Medical District - UIC West Campus / Rush Hospital",
    "1221": "Illinois Medical District - Ogden Ave / Damen Ave",
    "1222": "Tri-Taylor - Roosevelt Rd / Western Ave Crossroads",
    "1223": "University Village - Taylor St Historic Little Italy",
    "1224": "University Village - UIC East Campus / Halsted St",
    "1225": "University Village - Roosevelt Rd / Canal St Commercial Hub",
    "1231": "Pilsen - 18th St / Ashland Ave Arts & Cultural District",
    "1232": "Pilsen - 18th St / Halsted St Arts Corridor",
    "1233": "Pilsen - Cermak Rd / Blue Island Ave",
    "1234": "Pilsen - Cermak Rd / Damen Ave Industrial Corridor",
    "1235": "Heart of Chicago - 24th St / Oakley Ave Restaurant District",

    # District 14 - Shakespeare
    "1411": "Logan Square - Logan Blvd / Milwaukee Ave Monument",
    "1412": "Logan Square - Fullerton Ave / Kedzie Ave Blue Line",
    "1413": "Logan Square - Armitage Ave / Central Park Ave",
    "1414": "Logan Square - Wrightwood Ave / Kimball Ave",
    "1421": "Bucktown - Damen Ave / Armitage Ave Boutique Corridor",
    "1422": "Bucktown - Western Ave / Cortland St (The 606 Trail)",
    "1423": "Bucktown - Elston Ave / Webster Ave Commercial Gateway",
    "1424": "Wicker Park - Milwaukee / North / Damen Six-Corners Hub",
    "1431": "Humboldt Park East - Division St / Paseo Boricua Gateway",
    "1432": "Humboldt Park East - North Ave / California Ave",
    "1433": "Ukrainian Village - Chicago Ave / Damen Ave Historic Strip",
    "1434": "West Town - Grand Ave / Western Ave Industrial Arts",

    # District 15 - Austin
    "1511": "Austin - North Ave / Cicero Ave Retail Hub",
    "1512": "Austin - Division St / Central Ave Commercial",
    "1513": "Austin - Chicago Ave / Austin Blvd Gateway",
    "1514": "Austin - Chicago Ave / Laramie Ave",
    "1521": "Austin - Lake St / Central Ave Green Line Terminal",
    "1522": "Austin - Madison St / Austin Blvd",
    "1523": "Austin - Madison St / Central Ave Crossroads",
    "1524": "Austin - Jackson Blvd / Columbus Park Lagoon",
    "1531": "Austin - Harrison St / Cicero Ave I-290 Gateway",
    "1532": "Austin - Roosevelt Rd / Austin Blvd Border",
    "1533": "Galewood - Grand Ave / Central Ave Commercial",
    "1534": "Galewood - Narragansett Ave / North Ave",

    # District 16 - Jefferson Park & O'Hare
    "1611": "Jefferson Park - Milwaukee / Lawrence Transit Terminal",
    "1612": "Jefferson Park - Foster Ave / Central Ave",
    "1613": "Gladstone Park - Milwaukee Ave / Austin Ave",
    "1614": "Forest Glen - Elston Ave / Peterson Ave Expressway",
    "1621": "Norwood Park - Northwest Hwy / Harlem Ave Metra",
    "1622": "Norwood Park - Devon Ave / Nagle Ave",
    "1623": "Edison Park - Northwest Hwy / Touhy Ave Dining Strip",
    "1624": "Edison Park - Ozanam Ave / Touhy Ave Border",
    "1631": "Portage Park - Six-Corners (Milwaukee / Irving Park / Cicero)",
    "1632": "Portage Park - Montrose Ave / Cicero Ave Crossroads",
    "1633": "Dunning - Irving Park Rd / Harlem Ave Commercial",
    "1634": "Dunning - Addison St / Narragansett Ave",
    "1651": "O'Hare Airport - Main Terminal Core (Terminals 1, 2, 3 & ATS)",
    "1652": "O'Hare Airport - Bessie Coleman Dr / Cargo Logistics Corridor",
    "1653": "O'Hare Airport - Cumberland Ave / I-90 Transit Center",
    "1654": "O'Hare Airport - Mannheim Rd / Higgins Rd Hotel Plaza",

    # District 17 - Albany Park
    "1711": "Albany Park - Lawrence Ave / Kedzie Ave Global Dining",
    "1712": "Albany Park - Montrose Ave / Kimball Ave Brown Line",
    "1713": "Albany Park - Foster Ave / Pulaski Rd North Park Hub",
    "1721": "Irving Park - Irving Park Rd / Kedzie Ave Crossroads",
    "1722": "Irving Park - Addison St / Pulaski Rd",
    "1723": "Irving Park - Belmont Ave / Kimball Ave Blue Line",
    "1724": "Old Irving Park - Irving Park Rd / Kostner Ave Metra",
    "1731": "Avondale - Milwaukee Ave / Belmont Ave Polish Village",
    "1732": "Avondale - Elston Ave / Addison St Retail Corridor",
    "1733": "North Park - Bryn Mawr Ave / Kimball Ave NEIU Campus",
    "1734": "Ravenswood Manor - Manor Ave / Francisco Ave Brown Line",

    # District 18 - Near North
    "1811": "Old Town - Wells St / North Ave Historic Comedy Corridor",
    "1812": "Old Town - Sedgwick St / Division St",
    "1813": "Goose Island - Division St / Halsted St Tech & Brewery Hub",
    "1814": "Near North - Clybourn Ave / Halsted St Retail Corridor",
    "1821": "Gold Coast - Rush St / Division St Luxury Nightlife",
    "1822": "Gold Coast - Oak St / Lake Shore Dr Luxury Promenade",
    "1823": "River North - Chicago Ave / Franklin St Art Galleries",
    "1824": "River North - Ontario St / LaSalle St Entertainment",
    "1831": "River North - Merchandise Mart / Kinzie St Tech Hub",
    "1832": "Streeterville - Magnificent Mile (Michigan Ave & Chicago Ave)",
    "1833": "Streeterville - Northwestern Memorial Hospital Medical Core",
    "1834": "Streeterville - Navy Pier / Grand Ave Lakefront Promenade",

    # District 19 - Town Hall
    "1911": "Lincoln Park - Armitage Ave / Halsted St Historic Boutiques",
    "1912": "Lincoln Park - Fullerton Ave / DePaul University Campus",
    "1913": "Lincoln Park - Lincoln Park Zoo / Stockton Dr Promenade",
    "1914": "Lincoln Park - Clark St / Diversey Pkwy Gateway",
    "1915": "Sheffield Neighbors - Webster Ave / Sheffield Ave",
    "1921": "Lakeview - Belmont Ave / Clark St Red Line Transit Hub",
    "1922": "Northalsted - Halsted St / Addison St Entertainment Hub",
    "1923": "Lakeview East - Broadway / Belmont Ave Retail Corridor",
    "1924": "Wrigleyville - Clark St / Addison St (Wrigley Field Stadium)",
    "1925": "Lakeview - Ashland Ave / Belmont Ave Crossroads",
    "1931": "Roscoe Village - Belmont Ave / Damen Ave Dining Strip",
    "1932": "North Center - Lincoln / Irving Park / Damen Six-Corners",
    "1933": "Lakeview East - Sheridan Rd / Lake Shore Dr Promenade",
    "1934": "Lincoln Park - Fullerton Pkwy / Lake Shore Dr Beachfront",
    "1935": "Southport Corridor - Southport Ave / Grace St Retail Strip",

    # District 20 - Lincoln
    "2011": "Andersonville - Clark St / Foster Ave Swedish Heritage Strip",
    "2012": "Andersonville - Clark St / Bryn Mawr Ave Boutiques",
    "2013": "Edgewater - Bryn Mawr Ave / Broadway Red Line",
    "2021": "Lincoln Square - Lincoln Ave / Western Ave Plaza",
    "2022": "Lincoln Square - Lawrence Ave / Damen Ave Cultural Center",
    "2023": "Ravenswood - Montrose Ave / Damen Ave Industrial Arts",
    "2024": "Bowmanville - Bowmanville Ave / Western Ave",
    "2031": "West Ridge - Devon Ave South Asian Desi Corridor",
    "2032": "West Ridge - Touhy Ave / Western Ave Commercial Hub",
    "2033": "Edgewater Glen - Broadway / Devon Ave University Edge",

    # District 22 - Morgan Park
    "2211": "Beverly - 95th St / Western Ave Commercial Hub",
    "2212": "Beverly - 103rd St / Longwood Dr Historic Hill",
    "2213": "Beverly - 107th St / Walden Pkwy Metra",
    "2221": "Morgan Park - 111th St / Western Ave Commercial",
    "2222": "Morgan Park - 115th St / Vincennes Ave",
    "2223": "Morgan Park - 119th St / I-57 Expressway Gateway",
    "2231": "Mount Greenwood - 103rd St / Kedzie Ave Commercial",
    "2232": "Mount Greenwood - 111th St / Pulaski Rd Retail Hub",
    "2233": "Mount Greenwood - 115th St / Kedzie Ave",
    "2234": "Mount Greenwood - 111th St / Central Ave Gateway",

    # District 24 - Rogers Park
    "2411": "Rogers Park - Howard St / Clark St Transit Terminal",
    "2412": "Rogers Park - Touhy Ave / Clark St",
    "2413": "Rogers Park - Morse Ave / Glenwood Ave Arts District",
    "2421": "Rogers Park - Pratt Blvd / Sheridan Rd Lakefront Beach",
    "2422": "Rogers Park - Devon Ave / Broadway Loyola Lakefront Campus",
    "2423": "Rogers Park - Clark St / Devon Ave",
    "2431": "West Ridge - Touhy Ave / California Ave Commercial",
    "2432": "West Ridge - Peterson Ave / Lincoln Ave Modernist Strip",
    "2433": "West Ridge - Warren Park / Western Ave Recreation Hub",

    # District 25 - Grand Central
    "2511": "Belmont Cragin - Belmont Ave / Central Ave Commercial",
    "2512": "Belmont Cragin - Diversey Ave / Laramie Ave",
    "2513": "Belmont Cragin - Armitage Ave / Cicero Ave Industrial",
    "2514": "Belmont Cragin - Grand Ave / Fullerton Ave Retail Center",
    "2515": "Belmont Cragin - Fullerton Ave / Narragansett Ave",
    "2521": "Hermosa - Armitage Ave / Pulaski Rd (Walt Disney Birthplace)",
    "2522": "Hermosa - Fullerton Ave / Kostner Ave",
    "2523": "Kelvyn Park - Wrightwood Ave / Kostner Ave",
    "2524": "Kelvyn Park - Diversey Ave / Pulaski Rd Retail Strip",
    "2525": "Kelvyn Park - Belmont Ave / Pulaski Rd Crossroads",
    "2531": "Montclare - Grand Ave / Harlem Ave Brickyard Mall Hub",
    "2532": "Montclare - Fullerton Ave / Harlem Ave",
    "2533": "Galewood North - North Ave / Narragansett Ave",
    "2534": "Hanson Park - Grand Ave / Central Ave Metra Hub",
    "2535": "Cragin - Cicero Ave / Grand Ave Major Intersection",

    # Citywide Patrol
    "3100": "Citywide Expressway & Traffic Enforcement Command",
    "6100": "Citywide Tactical & Expressway Enforcement Unit",
    "61": "Chatham - 79th St / State St Commercial Gateway",
    "614": "Chatham - 87th St / Dan Ryan Expressway Commercial Hub",
    "726": "West Englewood - 71st St / Western Ave Transit Corridor",
    "733": "Englewood - 75th St / Halsted St Commercial Crossroads",
    "734": "Englewood - 76th St / Racine Ave",
    "735": "Englewood - 79th St / Halsted St Major Transit Hub",
    "1655": "O'Hare Airport - Multi-Modal Facility & Rental Car Concourse",
    "2424": "Rogers Park - Loyola University Lake Shore Campus & Sheridan Rd"
}


def resolve_chicago_zone_name(beat_id: str) -> Dict[str, str]:
    """
    Returns authentic Chicago neighborhood, district, and corridor details
    for any of the 275 Chicago Police Department beats.
    """
    cleaned = str(beat_id).strip().upper().replace("BEAT_", "").replace("BEAT", "")
    
    # 1. Exact Corridor Override Match
    if cleaned in EXACT_CHICAGO_BEAT_DETAILS:
        name = EXACT_CHICAGO_BEAT_DETAILS[cleaned]
    else:
        name = None

    # 2. Extract District Number from CPD numbering convention
    num_part = "".join(filter(str.isdigit, cleaned))
    dist_num = 1
    if len(num_part) >= 3:
        dist_num = int(num_part[:-2])
    elif len(num_part) > 0:
        dist_num = int(num_part[0])

    dist_profile = CPD_DISTRICT_PROFILES.get(dist_num, {
        "district": f"{dist_num:02d}th Police District",
        "neighborhood": f"Chicago Sector {cleaned}",
        "type": "Municipal Urban Corridor"
    })

    if not name:
        name = f"{dist_profile['neighborhood']} (Sector {cleaned})"

    return {
        "zone_id": cleaned,
        "name": name,
        "district": dist_profile["district"],
        "type": dist_profile["type"],
        "full_label": f"Beat {cleaned}: {name} [{dist_profile['district']}]"
    }
