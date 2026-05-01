"""
Name pool for character assignment in quiz question generation.

3,300+ names: ~1,650 first names + ~1,650 last names across diverse
cultural backgrounds. Deterministic hash-based assignment ensures:
  - Same question_id always gets the same characters
  - Different questions get different characters
  - 1,650 x 1,650 = 2,722,500 unique full-name combinations per role
  - Eliminates LLM name-bias (e.g. Dr. Navarro in 87% of questions)
"""

import hashlib

# ══════════════════════════════════════════════════════════════
# First Names (~1,650) — multi-ethnic, mixed gender
# ══════════════════════════════════════════════════════════════

FIRST_NAMES = tuple(sorted(set("""
Aaron Aaliyah Aanya Abdi Abdul Abel Abhishek Abigail Abraham Ada
Adaeze Adam Adama Adele Adeola Adi Adil Aditi Aditya Adrian
Adriana Adrienne Agnieszka Ahmad Ai Aidan Aimee Aisha Ajay Akash
Akemi Akiko Akira Akosua Alain Alana Alejandro Aleksei Alena Alessandra
Alessandro Alex Alexander Alexandra Alexei Alfonso Alfred Ali Alice Alicia
Alina Alison Alma Alok Alonzo Alvaro Amal Amalia Amanda Amara
Amelia Amin Amina Aminata Amir Amira Amit Amita Amy Ana
Anand Anastasia Anders Andrea Andrei Andres Andrew Andrzej Angel Angela
Angelo Anh Anika Anil Anita Anjali Anna Annika Anong Anthony
Antoine Antoinette Antonio Anya Aoife Aparna Ara Arash Archana Ariana
Ariel Arif Arjun Armando Arnaud Arthur Arun Arya Asha Ashley
Ashok Ashwin Asma Astrid Audrey Austin Autumn Ava Avery Avi
Avigail Ayaan Ayana Ayesha Ayodele Aysel Azadeh Aziz Azlan Babak
Bailey Bala Bao Barbara Baris Bartosz Beatrice Beatriz Belen Ben
Benjamin Bernard Beth Bethany Bhavna Bianca Bilal Binh Biruk Bjorn
Blair Blake Bo Bodhi Bola Bongani Boris Bradley Brandon Brayden
Brendan Brennan Brent Brian Bridget Brigitte Brittany Brooklyn Bruce Bruno
Bryan Budi Byron Caleb Cameron Camila Camille Carl Carla Carlos
Carmen Caroline Carter Casey Catalina Catherine Celine Cesar Chad Chandra
Charles Charlotte Chelsea Chen Cheng Chi Chidinma Chidi Chiamaka Chinelo
Chinwe Chris Christian Christina Christine Chukwu Ciara Cillian Claire Clara
Claudia Clayton Cleo Cliff Clinton Cody Colette Colin Colm Connor
Conor Corey Cormac Craig Cristina Cruz Crystal Curtis Cyrus Dagny
Dahlia Daichi Dalia Dalila Damian Dana Daniel Daniela Danish Dante
Daphne Daria Dariush Darnell Darren David Dawn Dean Deborah Declan
Deepa Deepak Deirdre Delia Denis Denise Dennis Derek Devon Devika
Dhruv Diana Diane Diego Dieter Dilip Dimitri Dinesh Divya Dolores
Dominic Dominique Donald Donna Dorian Dorota Dorothy Douglas Drake Drew
Dustin Dwayne Dylan Eamon Eduardo Edward Edwin Eileen Elaine Eleanor
Elena Elham Elif Elijah Elisa Elise Elizabeth Ellen Elliott Eloise
Elsa Emeka Emery Emi Emil Emilia Emilio Emily Emma Emmanuel
Emre Enrique Erika Erik Ernest Ernesto Esperanza Esther Ethan Etienne
Eugene Eva Evan Eve Evelyn Ewa Ezekiel Eze Fabian Fabienne
Fadil Farah Farhan Farid Farida Farzana Fatima Fatou Faye Fe
Federico Felipe Femi Feng Fiona Fletcher Flora Florence Folake Francis
Francisco Franco Frank Frankie Franklin Frederick Freya Friedrich Gabriel Gabriela
Galina Ganesh Gauri Gary Geeta Geneva Genevieve George Gerald Gerhard
Getachew Gideon Gillian Giovanni Giselle Gloria Gordon Grace Graham Grant
Gregory Greta Grzegorz Guadalupe Gudrun Guillaume Gunther Gustavo Gwen Habib
Hadi Hadiya Hajin Hakan Halima Hana Hang Hannah Hans Hao
Harold Harper Harrison Harry Harsh Haruki Hasan Hassan Hayden Hazel
Heather Heidi Helen Helena Helmut Henry Herbert Hien Hikari Hiroshi
Holly Hong Howard Hua Hugo Huong Hunter Husain Ibrahim Ichiro
Idris Ifeoma Ignacio Igor Ike Ilana Iman Imani Imran Indira
Ingrid Irene Iris Isaac Isabel Isha Ishaan Ismail Istvan Ivan
Ivy Jabari Jack Jackson Jacob Jacques Jaime Jake Jakub Jalil
Jamal Jamila James Jamie Jan Jana Jane Janet Janine Jared
Jarvis Jason Jay Jaya Jean Jelena Jenna Jennifer Jenny Jeremy
Jerome Jerry Jesse Jessica Jia Jill Jim Jimmy Jin Jing
Jisoo Joan Joanne Joaquin Jocelyn Joel Johan Johann Johanna John
Jon Jonathan Jordan Jorge Jose Josef Joseph Josephine Joshua Joy
Joyce Juan Juana Judith Julia Julian Julie Jun June Junior
Justin Jyoti Kai Kaia Kaito Kalani Kamala Kamila Kamol Kana
Karan Karen Karin Karim Karl Karla Karuna Katalin Kate Katherine
Katya Kavita Kay Kaylee Keiko Keith Kelly Kelvin Kemi Kemp
Kendall Kenji Kenneth Kenya Kenzo Kerry Kevin Khadija Khalid Khalil
Khanh Kian Kieran Kim Kimberly Kiran Kirk Kirsten Klaus Kofi
Koji Konrad Kostas Krishna Kristin Kristen Krzysztof Kumar Kurt Kwame
Kwesi Kyle Laia Laila Lakshmi Laleh Lalita Lana Lance Lan
Lara Larissa Lars Laura Lauren Lawrence Layla Leah Lee Leela
Lei Leila Lena Leo Leon Leona Leonard Leonel Leonor Lerato
Leslie Liam Lila Lillian Lily Lin Linda Lindiwe Ling Lisa
Lise Livy Liz Lloyd Logan Lois Long Lorena Lorenzo Lori
Lorraine Lottie Louis Louise Lourdes Luca Lucas Lucia Lucien Lucille
Lucy Ludwig Luis Luisa Luke Luna Luther Luz Lydia Lynn
Mackenzie Madan Madhavi Madison Mae Maggie Magnus Mahesh Mai Maia
Maika Maisie Majid Makeda Mala Malcolm Malee Malia Malik Malika
Mandisa Manish Manuel Mara Marc Marcel Marcela Marco Margaret Margaux
Maria Mariam Marian Marianne Marie Marisol Marissa Mark Marlene Marta
Martha Martin Mary Mason Mateo Mathieu Matilde Matthew Maureen Maya
Meena Meg Megan Mei Melissa Mercedes Meredith Meron Michael Michelle
Miguel Mikhail Mila Milan Mildred Miles Mina Ming Minjae Mira
Miranda Miriam Miroslav Mitchell Mohamed Monica Monika Monroe Morag Morgan
Morris Moshe Mpho Murat Mustafa Myra Nadia Nadine Nadira Naiara
Naledi Nam Nana Nandini Nancy Naomi Napat Naresh Narissa Nasim
Nasreen Nasser Natalia Natalie Natasha Nathan Nathaniel Neal Neelam Neha
Neil Nelson Neville Ngoc Ngozi Niamh Nicholas Nicole Nikhil Nikita
Nikolai Nils Nima Nina Nisha Njeri Noah Noel Noelle Nora
Norma Norman Noura Nuri Obinna Octavia Oleg Olga Oliver Olivia
Olumide Omar Ophelia Oren Orla Orlando Oscar Otieno Otto Owen
Pablo Padma Palesa Pamela Pankaj Paolo Parisa Parker Patricia Patrick
Paul Paula Paulina Pavan Pavel Pearl Pedro Peggy Pei Penelope
Percy Perla Perry Pete Peter Petra Philip Phoebe Phong Phuong
Pierce Pilar Ping Piotr Piper Pooja Pradeep Prakash Preston Priya
Priyanka Qian Quinn Rachael Rachel Radha Rafael Rafiq Rahul Raj
Rajesh Ralph Ramesh Rami Ramon Rana Randall Rani Rania Raphael
Rashid Raul Raven Ravi Raymond Rebecca Reese Reggie Regina Reina
Rekha Remi Ren Renata Rene Renee Reuben Rex Reza Rhea
Rhonda Ricardo Richard Rick Riku Rima Rina Rita River Robert
Robin Rochelle Rodrigo Roger Rohan Roland Roman Ronald Rory Rosa
Rosalie Rose Rosemary Ross Rowan Roy Roya Ruben Ruby Rudolf
Rui Russell Ruth Ryan Ryo Sabina Sachin Sadie Safa Sage
Sahar Said Saima Sakura Salim Salma Salman Salvador Salvatore Sam
Samir Samira Samuel Sana Sanaa Sandeep Sandra Sanjay Sanna Sapna
Sara Sarah Sarita Sasha Savita Sawyer Sayuri Scott Sean Sebastian
Selena Selin Selma Serena Sergio Seth Seung Shaina Shalini Shamim
Shana Shane Shannon Sharon Shaun Shawn Sheila Shelby Shelley Shelly
Shiloh Shin Shirin Shivani Shreya Shu Sibongile Siddharth Sidney Siegfried
Sienna Sierra Sigrid Silvia Simone Sipho Siri Sita Skyler Sofia
Sol Solomon Solveig Somchai Sondra Sonia Sonja Sora Soren Soraya
Sorcha Spencer Stacy Stanley Stella Stephan Stephanie Stephen Sterling Steve
Steven Stuart Suki Sullivan Sultana Sumitra Sunil Sunita Suresh Susan
Susana Sven Sveta Swati Sydney Sylvia Tabitha Tahira Taiwo Takeshi
Tamar Tamara Tammy Tanvi Tanya Tao Tara Tariq Tasha Tatiana
Taylor Tedros Teresa Terrence Terry Tessa Thabo Thai Thalia Thandeka
Thandiwe Thanh Thao Theodore Theresa Thierry Thomas Thu Thuy Tiffany
Timothy Tina Tobias Todd Tomas Tommy Tony Tori Toyin Tracy
Travis Trevor Tricia Tristan Troy Trudy Tucker Tunde Tyler Uma
Umar Ursula Usha Valentina Valerie Valeria Van Vanessa Varun Vasili
Vera Veronica Veronique Vicente Victor Victoria Vidya Vijay Vikram Vince
Vincent Viola Violet Virgil Virginia Vivek Vivian Vladimir Wade Walter
Wanda Warren Wayne Wei Wen Wendell Wendy Wesley Whitney Wilbur
William Winston Wolf Wolfgang Woody Wyatt Xander Xavier Xia Xin
Xu Yael Yamileth Yan Yang Yara Yash Yasir Yasmin Yolanda
Yong Yosef Youssef Yu Yuan Yui Yuki Yumi Yusuf Yvette
Yvonne Zahra Zaid Zainab Zara Zeke Zelda Zena Zev Zhen
Zhi Zinhle Zoe Zofia Zola Zora Zuri
Addison Adelaide Adeline Agnes Aiden Ainsley Ajani Akane Aksel Alaina
Albert Alexa Alfie Allegra Allen Alondra Althea Alvin Amani Amber
Amelie Angus Annabel Ansel Antonia Arabella Arden Arlo Armand Athena
Atlas August Aurora Axel Basil Baxter Bea Beckett Bella Benedict
Benita Benny Bernadette Bernardo Bertrand Beverly Billy Blaine Blaise Blythe
Bobby Bonnie Bram Braxton Brenda Briana Brielle Brody Brooke Bryson
Cadence Caitlin Callum Camden Cara Carissa Carlton Carmela Carol Carolyn
Cassandra Cassidy Cecilia Cedric Charity Chloe Clarence Clarissa Claudette Clemence
Clyde Colby Colleen Comfort Conrad Constance Coraline Cordelia Cornelius Courtney
Crispin Cullen Cynthia Cyril Dagmar Dale Dalton Damaris Damon Dane
Dani Darcy Darin Darla Darwin Davon Dawson Debra Della Demetrius
Desmond Destiny Devin Digby Dillon Dina Dion Dirk Dolly Donato
Dora Doreen Dragan Dulce Earl Ebba Eddie Edgar Edith Edmund
Edna Edris Edwin Elara Eldon Eliana Elias Elio Eliza Ellery
Elmer Elora Elton Elvira Ember Emerson Enid Ephraim Errol Esme
Esmeralda Estelle Etta Evangeline Everett Ezra Faisal Fallon Farrah Felicia
Felix Fergus Fidel Finn Fiorella Fleur Flint Flora Florence Floyd
Ford Forrest Frances Francisca Fraser Fritz Gael Gaia Galen Gareth
Garrett Gavin Gemma Gene Geoffrey Gerard Gertrude Gianna Gilda Giles
Gina Giovanna Gladys Glen Glenn Goldie Graeme Griselda Grover Gus
Hadley Hailey Hamza Hank Harlan Harley Harmony Harriet Hartley Hattie
Haven Hawk Hayley Hector Heloise Hendrix Herb Herman Hester Holden
Homer Hope Horace Ida Iggy Ilya Imogen India Ines Ingvar
Ion Ira Irena Irwin Isabell Isadora Isolde Israel Iva Ivana
Jace Jacinda Jacqueline Jade Jalen Jameson Janelle Janis Jared Jasmine
Jasper Jed Jelani Jemma Jensen Jewel Jimena Jinx Jodie Jolene
Jonah Josefina Josiah Judd Jules Juliet Juniper Justice Justine Kade
Kaleb Kamari Kamilah Kane Kara Karel Karla Kasim Kassandra Kathryn
Katie Kayla Keanu Keegan Kellan Kelsey Kendrick Keoni Kerry Keshav
Keva Khari Kiki Killian Kingston Kira Kit Kobe Kora Kosta
Kris Kristian Kyra Lacey Lamar Landon Langston Larissa Larry Latasha
Laurel Leander Leanne Leigh Lennox Leopold Lester Leta Leticia Levi
Lexie Lilah Lilian Lincoln Lisette Llewellyn Lola London Lonnie Loren
Loretta Lorna Lotte Lotus Louie Lucinda Luka Lukas Lyra Mabel
Madeline Magdalena Magnolia Major Mallory Malone Mamie Manny Marcelino Margo
Margot Marguerite Maribel Marilyn Marjorie Marlena Marley Marquis Martina Marvin
Matias Matthias Maude Maurice Max Maxim Maximilian Maxine Meadow Meera
Melanie Melody Merle Micah Mika Millicent Milo Milton Mimi Minerva
Minnie Mirabel Moira Molly Monte Monty Moses Myra Myrna Myrtle
Nadim Naia Najib Nakita Napoleon Nash Naya Nehemiah Nell Nellie
Nestor Neva Nia Niall Nico Nicolette Nile Nola Norbert Noreen
Nova Oakley Oberon Octavio Odessa Odette Olive Olympia Onyx Opal
Orion Orson Oswald Otis Paige Pandora Pasha Paulette Payton Penny
Percival Peregrine Peyton Philomena Pieter Placido Poppy Prescott Primrose Prince
Priscilla Prudence Quentin Quill Quincy Rachelle Rafferty Raina Raleigh Ramona
Ransom Raoul Rasheed Raylan Reagan Rex Rhett Rhiannon Richie Ridge
Rigel Roberta Rocco Rocky Roderick Rodney Rolando Romina Ronin Rosalind
Roscoe Rosetta Roxanne Rudy Rufus Rupert Ruslan Ruthie Sabrina Sage
Sahil Saint Saira Salome Samson Sandy Santino Saoirse Savannah Scarlett
Scout Selene Serge Seymour Shakira Shantel Sharpe Shay Sheldon Shepherd
Sheridan Sherlock Sherry Shirley Sibyl Silas Simon Skip Sloane Socorro
Soleil Sparrow Spike Star Stellan Storm Sully Summer Sunniva Sybil
Sylvester Tabitha Tad Tahlia Talon Tamika Tatum Tavish Teagan Tegan
Tennyson Terrell Thaddeus Thelma Theron Thora Thurston Tiana Tiberius Tillie
Toby Tomoko Topaz Torben Torin Trace Trent Trenton Trey Trilby
Trinity Tudor Twyla Tycho Tyra Tyrell Tyson Ulric Umberto Uriel
Valentino Vance Vasiliki Vaughn Veda Verena Verna Vesper Vida Vilma
Viviana Waldo Willa Willard Wilma Winona Wren Wyatt Wynne Xanthe
Xiomara Yamila Yancy Yaroslav Yasmine Yosef York Zachariah Zander Zane
Zenobia Zephyr Zion Zita Zubin Zula
""".split())))

# ══════════════════════════════════════════════════════════════
# Last Names (~1,650) — multi-ethnic
# ══════════════════════════════════════════════════════════════

LAST_NAMES = tuple(sorted(set("""
Abbott Abdallah Abdulrahman Abe Abiodun Abraham Abramov Acosta Adachi Adams
Adebayo Adekunle Adeyemi Afolabi Agarwal Aguilar Aguilera Ahmad Ahmed Ahn
Ajayi Akhtar Akinola Akinyemi Al-Farsi Al-Hamad Al-Hassan Al-Rashid Alarcon
Alavi Aldridge Alexander Ali Allen Almeida Alvarado Alvarez Amadi Amato
Amin Amiri Anand Anderson Andersen Angelopoulos Antonescu Aoki Aquino Archer
Arias Arora Armstrong Arnold Arslan Asante Aslam Athanasiou Atkinson Austin
Avila Aydin Babic Badawi Bae Baek Bahrami Bailey Baker Baldwin
Balogun Balog Banda Banerjee Banks Bao Barker Barnes Barrett Barrera
Barry Bartlett Basu Batista Bates Bauer Bautista Becker Begum Belov
Beltran Ben-David Benavides Benedict Bennett Benson Berg Bergman Bernard Berry
Bhatt Bhattacharya Bianchi Bishop Black Blackwell Blair Blake Blanc Blanchard
Blanco Bloom Bogdan Bolton Bond Bonner Booth Boucher Boyd Boyle
Bradley Brady Brand Braun Brennan Briggs Brock Brooks Brown Bruno
Bryant Buchanan Buck Buckley Bui Burke Burns Burton Bush Butler
Byrne Cabrera Cai Calderon Caldwell Cameron Campbell Campos Cannon Cao
Cardenas Carey Carlson Carlton Carmichael Carpenter Carr Carrillo Carroll Carson
Carter Castillo Castro Cervantes Chadha Chakraborty Chambers Chan Chandler Chang
Chapman Chatterjee Chavez Chen Cheng Chevalier Choi Chopra Choudhury Chow
Christensen Chung Clark Clayton Clement Cole Coleman Collins Colombo Conley
Connolly Connor Contreras Conway Cook Cooper Cordero Cornwell Cortes Costa
Cox Craig Crawford Croft Cross Cruz Cui Cunningham Curtis Dahl
Dalton Daly Daniels Darwish Das Dasgupta Davidson Davies Davis Davari
Dean Delaney Delgado Demirel Demir Dennis Deng Desai Deschamps Deshpande
Devlin Dewi Dey Dhillon Diallo Diaz Ding Dixon Dlamini Dmitriev
Do Dobrescu Doherty Dong Donnelly Donovan Doyle Drake Du Dube
Dubois Dumont Duncan Dunlap Dunn Dupont Durand Duran Duval Dvorak
Edwards Ekwensi Elliott Ellis Emerson Endo Erdogan Eriksen Escobar Espinoza
Esposito Estrada Evans Everett Faber Fahey Fakhri Fallon Fan Fang
Farkas Farmer Farouk Farrell Faulkner Fedorov Feng Fernandez Ferrara Ferrari
Fields Figueroa Finch Finn Fischer Fisher Fitzgerald Fitzpatrick Fleming Fletcher
Flores Flynn Fontaine Forbes Ford Forsberg Forsyth Foster Fournier Fox
Francis Franco Frank Franklin Fraser Freeman Friedman Frost Fu Fuentes
Fujimoto Fujita Fuller Gagne Gaines Gallagher Gallegos Gallo Galvan Gandhi
Gao Garcia Gardner Garnier Garrett Garza Gebremedhin George Georgiadis Ghasemi
Ghosh Gibbons Gibson Gilbert Gill Gillespie Girard Glenn Golzar Gomez
Gonzalez Goodman Goodwin Gordon Goswami Graham Grant Graves Gray Green
Greenwood Greer Gregory Gresham Griffin Griffith Grimes Gross Gruber Gu
Guerra Guerrero Guevara Gulati Gul Guo Gupta Gustafsson Gutierrez Guzman
Ha Habib Haddad Hadid Hagerty Haider Hajek Hale Hall Halvorsen
Hamilton Hammond Han Hancock Hanna Hansen Hao Hara Hardy Harper
Harmon Harris Harrison Hart Hartmann Harvey Hasan Hasegawa Hashim Hashemi
Hassan Haugen Hawkins Hayashi Hayes Hayward He Heath Henderson Henriksen
Henry Herman Hernandez Herrera Hewitt Hicks Higgins Hill Ho Hoang
Hobbs Hodge Hoffman Hofmann Holland Holloway Holmes Holt Hong Hopkins
Horak Horton Horvath Hosseini Hou Howard Howell Hsu Hu Huang
Hubbard Huber Hudson Hughes Humphrey Hunt Hunter Hurtado Hussain Hwang
Hyde Ibrahim Igwe Ikeda Ingram Inoue Ionescu Iqbal Irving Ismail
Ito Ivanov Iverson Jackson Jacobs Jacobsen Jafari Jain James Jang
Jankowski Jarvis Javed Jefferson Jenkins Jensen Jeon Jiang Jimenez Jin
Jo Johansson Johansen Johnson Johnston Jones Jordan Jorge Joshi Joyce
Juarez Jung Kaiser Kamara Kaminski Kang Kapoor Karim Karlsson Kato
Kaufman Kaur Kavanaugh Kawaguchi Kay Kaya Kazemi Keating Keller Kelly
Kemp Kendall Kennedy Kent Kerr Khalil Khan Khatri Khumalo Kim
Kimani Kimura King Kirk Kirkpatrick Klein Knapp Knight Knox Ko
Kobayashi Koch Kofi Kohli Kolar Kone Konrad Kowalski Kozlov Kramer
Krishnan Kristiansen Kuhn Kulkarni Kumar Kuznetsov Kwon Lacroix Laing Lam
Lambert Lancaster Lane Lang Lara Larsen Larson Lau Laurent Lawson
Le Leal Lebedev Leclerc Lee Lefebvre Legrand Lei Leon Leonard
Leong Lerner Leslie Leung Levi Levine Lewis Li Liang Lim
Lima Lin Lindgren Liu Lloyd Long Lopez Lowe Lu Luo
Luther Lynch Lyons Ma Mabaso Machado Macias Mack Mackenzie Maddox
Maeda Mahajan Mahlangu Mahmoud Mahoney Mai Maina Maier Mak Makris
Malik Malone Maldonado Mancini Mandela Manning Manzo Marchetti Marin Marino
Markovic Marquez Marsh Marshall Martin Martinez Martini Martinelli Mason Mata
Mathur Matsuda Matsumoto Matthews Maxwell May Mayer Maynard Mbeki McCarthy
McCormick McDonald McGee McGrath McGuire McKay McKenna McKenzie McLean McMahon
Medina Mejia Melnyk Mendez Mendoza Menon Mensah Mercado Mercer Mercier
Merton Meyer Michel Middleton Miles Miller Mills Min Miranda Mirza
Mishra Mitchell Mittal Miyamoto Mochizuki Moffat Mohamed Mohan Mohammadi Molina
Monroe Montague Montoya Moody Moon Moore Morales Moreau Moreno Morgan
Mori Morita Morozov Morris Morrison Morse Moseley Mosley Moss Motsepe
Mueller Mukherjee Muller Munoz Murakami Murphy Murray Musa Musgrove Mustafa
Myers Mwangi Nagai Nagata Nagy Naidoo Nair Nakamura Nakano Nam
Nanda Narang Nash Nasser Navarro Nazari Ndlovu Neal Ndiaye Nelson
Nemec Nemeth Newman Newton Nguyen Nichols Nikitin Nikolic Nikolaou Nilsen
Nishida Nishimura Njoroge Nkosi Noble Nolan Norris Norton Novak Novikov
Nowak Nwosu Nygaard Obi Obrien Ochieng Ochoa Oconnell Oconnor Odonnell
Ogawa Ogunyemi Oh Okada Okamoto Okafor Okeke Okonkwo Oladipo Olaniyan
Oleary Olsen Olszewski Oniell Ono Orozco Ortega Ortiz Osei Osborne
Osullivan Otieno Owens Oyedele Ozturk Pace Padilla Page Pak Palacio
Palmer Pan Pandey Papadopoulos Papageorgiou Park Parker Parra Parsons Patel
Patil Patterson Pavlov Payne Pearce Pearson Pedersen Pellegrini Pena Peng
Peralta Perez Perkins Perry Petersen Peterson Petrescu Petrov Petrovic Pham
Phan Phillips Pierce Pike Pinder Pineda Platt Pollard Ponce Poole
Pope Popov Porter Portillo Posner Powell Powers Prasad Pratt Preston
Price Proctor Pruitt Pu Purnomo Qian Qin Qiu Quintero Quinn
Qureshi Radu Rahim Rahman Rai Rajan Ramirez Ramos Ramsey Rana
Randall Rankin Rao Rasmussen Rathore Rauf Rawlings Ray Raymond Raza
Reddy Reed Reeves Reid Reilly Ren Rendon Reyes Reynolds Rhodes
Ricci Rice Rich Richards Richardson Richter Ridley Riley Rinaldi Rios
Rivas Rivera Robbins Roberts Robertson Robinson Robles Rodriguez Rogers Rojas
Romano Romero Rose Rosenthal Ross Rossi Roth Rousseau Rowe Rowley
Roy Rubio Ruiz Rush Russell Russo Ryan Ryu Saadi Sabbagh
Sade Sadeghi Sagar Sahin Saito Said Saini Sakamoto Salazar Salem
Salinas Sampson Sanchez Sanders Sandoval Sandberg Santos Sanogo Sarin Sarpong
Sasaki Sato Savage Sawyer Saxena Schaefer Schmidt Schneider Scholz Schubert
Schulz Schwartz Scott Seifert Sen Seo Serrano Sethi Setiawan Shah
Shaheen Sharif Sharma Shaw Shelton Shen Shepherd Sherman Shi Sheth
Shields Shimizu Shin Shukla Siddiqui Sierra Silva Simmons Simon Simpson
Sinclair Singh Sinha Slater Small Smirnov Smith Snyder Sokolov Solis
Solomon Song Sosa Soto Sow Spencer Srinivasan Stafford Stanley Stanescu
Steele Stein Steiner Sterling Stevens Stevenson Stewart Stockton Stokes Stone
Strickland Stuart Su Suarez Sugiyama Sullivan Sun Sutton Suzuki Svensson
Svenson Swift Szabo Tabatabai Taguchi Takahashi Takeda Talbot Tan Tanaka
Tang Tanner Tao Tapia Tariq Tate Tavakoli Taylor Teixeira Tellez
Templeton Teng Thakur Thayer Thomas Thompson Thornton Tian Tierney Tiwari
Todd Toledo Tolstoy Torres Toure Townsend Tran Traore Trujillo Truong
Tsai Tucker Tung Turner Tuttle Ueda Underwood Uribe Usman Uzoma
Valdez Valencia Valenzuela Vance Vandenberg Vargas Varma Vasquez Vaughn Vega
Velasco Velasquez Venkatesh Vera Verma Vidal Villa Villanueva Villarreal Vincent
Volkov Vu Wade Wagner Wakefield Walker Wallace Walsh Walters Walton
Wambui Wang Ward Warner Warren Washington Watanabe Waters Watson Watts
Webb Weber Webster Welch Wells Werner West Westbrook Wheeler White
Whitfield Whitney Wibowo Wilder Wilkins Wilkinson Williams Willis Wilson Winters
Wolf Wolfe Wong Wood Woodward Wright Wu Xia Xiao Xie
Xu Xue Yadav Yamada Yamaguchi Yamamoto Yan Yang Yao Yates
Ye Yeboah Yildirim Yilmaz Yin Yoo Yoon Yossef Young Yu
Yuan Yue Yun Yusuf Zakharov Zambrano Zamora Zapata Zavala Zayed
Zeng Zhang Zhao Zheng Zhou Zhu Ziegler Zielinski Zimmerman Zulu
Abbot Addington Adler Agnew Aldridge Allison Almond Ambrose Amundsen Andrade
Appleby Archer Archibald Ashby Ashford Ashton Atwood Baines Ballard Bancroft
Bannister Barlow Barnett Barton Bassett Bateman Baxter Beaumont Becerra Bedford
Bell Bellamy Bello Bender Bergstrom Bernal Birch Blackburn Blackmore Blackwood
Blanton Block Blythe Boone Bowers Bowling Boyce Bradshaw Branch Brandt
Bray Brewster Bright Brock Brogan Bronson Brophy Broughton Bruce Bruner
Bryan Bryson Buckingham Buford Bull Bullock Burch Burgess Burris Burt
Cade Cain Callahan Calvert Cantu Cardoso Carlisle Carlton Carney Carrington
Caruso Caulfield Cavendish Cecil Chamberlain Chandra Chaney Chapel Cheung Chilton
Chin Church Clancy Clarkson Clay Cleary Clifford Cloud Coburn Coffey
Colton Combs Compton Corcoran Cornish Corrigan Cosgrove Cote Cotton Coulter
Courtney Coward Crabtree Crane Crenshaw Crews Croft Cromwell Crosby Crowell
Cullen Cummings Curry Cushing Daley Dalrymple Daly Dana Dane Darby
Dare Darnell Dashwood Davenport David Dear Decker Delany Delgadillo Dempsey
Devereaux Devine Dickinson Dillard Dillon Dodge Dolan Dorsey Dowling Downey
Drummond Duff Duggan Duke Dunbar Dunham Dunn Dutton Dwyer Eagleton
Earle Eastman Eaton Eckert Egerton Eldridge Elkins Elm Elmore Emery
Endicott English Ennis Enriquez Estes Ethridge Faber Fairbanks Fairchild Falcon
Falk Fanning Farley Farnsworth Farrow Feldman Fenwick Ferry Finch Finley
Flaherty Flanagan Flint Flora Flowers Fogarty Foley Fong Font Foreman
Fortune Fowler Frampton Franco Frasier Frazier Frost Fry Fuchs Fulbright
Fulton Gagnon Gale Galindo Galvin Gamble Gantt Garland Garnett Gatewood
Gauthier Gear Gentry Geyer Gibbins Gifford Gilchrist Gilman Gilmore Gladstone
Glasgow Gleason Glover Godfrey Goldberg Goldman Goldstein Goodrich Goodwin Gorman
Gould Grady Grafton Grantham Greenfield Gregor Griggs Grimm Grisham Grogan
Guthrie Haas Haddon Hagerman Haight Haines Haldane Haller Halsey Halstead
Hamblin Hamlin Hamrick Hanley Harding Hardwick Hare Hargrove Harlan Harlow
Harmon Harrell Harrington Hartley Hartwell Hastings Hathaway Hatton Havens Hawley
Hayden Haywood Healy Heaney Hedge Helms Hemingway Henson Herring Hewitt
Hidalgo Highsmith Hilliard Hillman Hines Hirsch Hobart Hobson Hodge Hoffman
Holbrook Holden Holliday Hollis Holman Holroyd Honeycutt Hood Hooks Hooper
Hoover Hoskins Houghton Howland Huff Hugo Hulbert Humble Hume Hunter
Huntley Hurd Hurley Hutchinson Hutton Hyland Hynes Inaba Ingle Ingraham
Innis Ivory Ives Ivory Jacks Janssen Jardine Jarrett Jasper Jeffries
Jett Jewett Jimenez Joachim Jolly Judge Kahn Kang Kasper Keane
Kearns Keating Keith Kellogg Kelton Kempf Kenyon Kershaw Kidd Kilgore
Kimball Kincaid Kinney Kirby Kirk Kirkland Kirkwood Kitchens Klug Knapp
Koenig Kolb Kraft Krause Kruger Kuhn Lacey Lachlan Ladd Lafayette
Laing Lake Lamb Landis Langford Langley Langston Lanham Lanier Lansing
Lark Latimer Lattimore Laughlin Lavigne Lawler Lay Leach Leal Ledger
Leftwich Lehman Leigh Lennon Lester Levitt Liddell Light Lilly Lindahl
Lindsey Linton Livingston Locke Lockhart Lofton Logue Lombard Lombardi Lord
Lovell Lovett Lowry Ludlow Lugo Lund Lundberg Lutz Lyndon Lytle
Mabry Mace Macklin Madden Maguire Maitland Major Mallory Malone Manor
Mansfield Manson Maple Marchand Markham Marlow Marquardt Marsden Marston Mathis
Maynard McAllister McBride McCaffrey McCann McClellan McCoy McCullough McFarland McGee
McIntosh McIntyre McKee McManus McMullen McNair McNeil McPherson Meade Mears
Medeiros Melton Menard Merrill Metcalf Middleton Milburn Millard Milne Minter
Moberg Mock Moffett Molloy Monaghan Monk Mooney Moorhouse Morley Morrow
Mortimer Morton Mosely Mullen Mulligan Munson Murdoch Muse Nagel Nance
Naples Napier Neale Neely Neff Neill Nesbitt Nestor Nevins Newcomb
Newell Newhouse Newsome Nicholson Niles Nix Noble Noel Nolan Noonan
Nordstrom Norwood Novotny Nugent Nye Oakes Oakland Oakwood Odom Ogden
Oldham Oleary Oliphant Oneal Orth Oswald Overstreet Pack Padgett Paine
Palmieri Parham Parish Park Parkinson Parrish Partridge Pascual Patch Paul
Paxton Peake Peck Pelham Pendleton Penn Pepper Perkins Peterman Petty
Pfeiffer Pham Pickering Pierson Pinkerton Pittman Platt Plummer Plunkett Pollard
Pollock Poole Portman Post Prater Prescott Prichard Priddy Prince Proctor
Purdy Putnam Quigley Quinlan Quirke Raines Rainey Ralston Ramsey Ramsay
Rand Randolph Ransom Ratliff Rawlins Ray Rayburn Rector Redmond Reece
Regan Remington Renner Rhoades Riddle Ridgeway Rigby Riggins Ring Ritchie
Riviera Roach Robb Rochester Rockwell Rodgers Rollins Roper Rosen Rountree
Rudd Runyon Rush Rutherford Rutledge Sacco Sadler Sage Salter Sanderson
Sangster Sargent Saville Scanlon Schiller Schreiber Schultz Scofield Seager Seaton
Selby Sellers Semple Seton Settle Seward Sewell Sexton Shackleton Shaffer
Shanahan Shapiro Shay Sheehan Shelby Sheriff Sherrill Shields Shipley Shore
Short Shultz Siddall Sikes Silverman Simonds Singer Skelton Skinner Sloan
Smallwood Smyth Snead Snell Snow Snowden Somers Sommer Sorensen Southgate
Sparrow Spaulding Spears Spence Springer Stafford Stamford Stanton Stark Starling
Starr Steed Steele Steinberg Stetson Stinson Stock Stoddard Stoner Storm
Stout Strange Stratton Stroud Strunk Stubbs Sturgis Suggs Summerfield Sunderland
Sutcliffe Swain Swanson Sweetman Sykes Taggart Talbot Talley Talmadge Tate
Temple Tennant Thayer Thiel Thorn Thornberry Thurman Tilden Tilley Tobin
Tolbert Tolman Tompkins Torrance Towers Trammel Trask Trotter True Trumbull
Tuck Tudor Tully Turnbull Tuttle Tyner Tyson Ulrich Urban Vale
Vandermeer Vann Varela Vega Venable Vickers Vidal Villalobos Vogel Voorhees
Wainwright Wake Walden Waldron Waller Wallis Warden Warfield Washington Waterhouse
Waterman Waugh Weathers Weller Wellman Wentworth Werner Westfall Weston Whitaker
Whitcomb Whitehead Whitley Whitmore Whittaker Wickham Wiggins Wilder Willard Wilmot
Windham Winkler Winslow Wise Withers Witt Wolcott Wolff Woodley Woodruff
Woolley Worth Wren Wyatt Wylie Wynn Yager Yale Yarborough Yeager
Yoder Yorke Youngblood Zamora Zavala Zeller Ziegler Zorn Zuniga Zurita
""".split())))


# ══════════════════════════════════════════════════════════════
# Clinical Settings (25 options)
# ══════════════════════════════════════════════════════════════

SETTINGS = (
    "outpatient therapy office",
    "community mental health center",
    "university counseling center",
    "private practice",
    "school-based counseling office",
    "hospital psychiatric unit",
    "rehabilitation center",
    "group therapy room",
    "pediatric clinic",
    "VA medical center",
    "forensic evaluation office",
    "substance abuse treatment center",
    "employee assistance program office",
    "residential treatment facility",
    "crisis intervention center",
    "telehealth session",
    "neuropsychology clinic",
    "primary care behavioral health office",
    "juvenile detention center",
    "geriatric care facility",
    "family therapy center",
    "college student health center",
    "correctional facility",
    "child development clinic",
    "integrated care clinic",
)


# ══════════════════════════════════════════════════════════════
# Deterministic Character Assignment
# ══════════════════════════════════════════════════════════════

def get_character_assignment(question_id, context="adult"):
    """Deterministic character assignment from question_id hash.

    Args:
        question_id: Unique question identifier
            (e.g., "BPSY-sleep-architecture-T2-v1")
        context: Age context — "adult" (default), "child",
            "adolescent", "older_adult", "young_adult"

    Returns:
        dict with clinician_name, client_name, client_first,
        client_age, setting
    """
    digest = hashlib.sha256(question_id.encode()).hexdigest()

    # Independent hash slices for each dimension
    clin_idx   = int(digest[0:4], 16) % len(LAST_NAMES)
    cli_first  = int(digest[4:8], 16) % len(FIRST_NAMES)
    cli_last   = int(digest[8:12], 16) % len(LAST_NAMES)
    age_seed   = int(digest[12:16], 16)
    setting_i  = int(digest[16:18], 16) % len(SETTINGS)

    # Ensure clinician and client don't share a last name
    if cli_last == clin_idx:
        cli_last = (cli_last + 1) % len(LAST_NAMES)

    # Age by context
    age_ranges = {
        "adult": (25, 62),
        "child": (6, 12),
        "adolescent": (13, 17),
        "older_adult": (65, 85),
        "young_adult": (18, 24),
    }
    lo, hi = age_ranges.get(context, (25, 62))
    age = lo + (age_seed % (hi - lo + 1))

    return {
        "clinician_name": f"Dr. {LAST_NAMES[clin_idx]}",
        "client_name": f"{FIRST_NAMES[cli_first]} {LAST_NAMES[cli_last]}",
        "client_first": FIRST_NAMES[cli_first],
        "client_age": age,
        "setting": SETTINGS[setting_i],
    }


def build_character_block(assignment, stem_pattern=None):
    """Format character assignment for injection into the user prompt.

    For vignette/scenario stems: names are mandatory.
    For conceptual stems: names are required IF the question uses characters.
    """
    names_block = (
        f"- Clinician: {assignment['clinician_name']}\n"
        f"- Client: {assignment['client_name']}, age {assignment['client_age']}\n"
        f"- Setting: {assignment['setting']}"
    )

    if stem_pattern in ("clinical_vignette", "scenario_completion"):
        return (
            f"\n## Character Assignment (MANDATORY -- use these exact names)\n"
            f"{names_block}\n"
            f"Do NOT invent other character names. Use ONLY the assigned names above "
            "in your question stem and options."
        )
    else:
        return (
            f"\n## Character Names (use if your question includes people)\n"
            f"{names_block}\n"
            f"If your question references a clinician, researcher, or client by name, "
            "you MUST use the names above. Do NOT invent your own character names."
        )
