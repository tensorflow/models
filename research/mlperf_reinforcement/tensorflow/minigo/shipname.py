# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import re
import petname

import go

MODEL_NUM_REGEX = "^\d{6}"
MODEL_NAME_REGEX = "^\d{6}(-\w+)+"


def generate(model_num):
    if model_num == 0:
        new_name = 'bootstrap'
    elif go.N == 19:
        new_name = random.choice(NAMES)
    else:
        new_name = petname.generate()
    full_name = "%06d-%s" % (model_num, new_name)
    return full_name


def detect_model_num(string):
    """Takes a string related to a model name and extract its model number.

    For example:
        '000000-bootstrap.index' => 0
    """
    match = re.match(MODEL_NUM_REGEX, string)
    if match:
        return int(match.group())
    else:
        return None


def detect_model_name(string):
    """Takes a string related to a model name and extract its model name.

    For example:
        '000000-bootstrap.index' => '000000-bootstrap'
    """
    match = re.match(MODEL_NAME_REGEX, string)
    if match:
        return match.group()
    else:
        return None


NAMES = """
ace
achates
acheron
achilles
aeneas
affray
agamemnon
ajax
alaric
alcide
alderney
alkmaar
alliance
ambush
amphion
anchorite
andrew
antelope
arachne
archer
ariel
artemis
artful
astute
athenienne
atlas
audacious
auriga
aurochs
batavier
belleisle
bellerophon
black-prince
brave
bulldog
bulwark
caesar
canopus
carnatic
cato
centaur
centurion
ceres
clove-tree
colossus
conqueror
constant-warwick
content
cormorant
coronation
courageux
crown
culloden
dauntless
defence
defiance
delfe
devastation
diadem
diligent
diomede
dispatch
dragon
dreadnought
druid
duchess
duke
eagle
elephant
enterprise
europa
exeter
expedition
fame
favorite
formidable
fortitude
foudroyant
fougueux
fury
ganges
gibraltar
glorieux
glory
golden-horse
golden-lion
golden-phoenix
goliath
grief
half-moon
hannibal
hastings
hawk
hawke
hazardous
hecate
hector
hercules
hero
heron
hood
hope
hotspur
hydra
icarus
illustrious
implacable
impregnable
indefatigable
indus
inflexible
intrepid
invincible
irresistible
isis
jupiter
juste
kent
kingfisher
kite
leander
leopard
leviathan
lightning
lincoln
lion
lively
lynx
magnanime
magnificent
majestic
malabar
marigold
mars
medusa
merlin
mermaid
minden
minotaur
moderate
modeste
monarch
mordaunt
namur
nassau
neptune
oberon
ocelot
odin
olympus
onslaught
onyx
opossum
opportune
oracle
orestes
orion
orpheus
osiris
otter
pallas
pandora
parthian
penelope
perseus
phoenix
portia
poseidon
prometheus
prospero
proteus
redoubtable
renown
repulse
research
resistance
revenge
robust
royal-sovereign
san-josef
san-nicolas
sans-pareil
sapphire
seagull
spartiate
sultan
superb
swift
swiftsure
temeraire
theseus
thunderer
tiger
trafalgar
tremendous
trident
triton
triumph
trusty
two-lions
tyger
unity
valiant
vanguard
venerable
vengeance
veteran
victorious
victory
vigilant
vindictive
waakzaamheid
wanderer
warrior
warspite
warwick
wasp
waterloo
wolf
zealous
zephyr
""".strip(' \t\n\r').split('\n')
