
AUDIO_RAG_TEXT = [
    [ # angry
        'I\'m so sick and tired of your constant complaining!',
        'How dare you speak to me in that condescending tone?',
        'I\'ve had it up to here with your laziness and incompetence.',
        'I\'m seething with rage over this injustice.',
        'You\'ve gone too far this time, and I won\'t let you get away with it.',
    ],
    [ # disgusted
        'The stench emanating from that dumpster is nauseating.',
        'I can\'t believe they served us this rotten, maggot-infested food.',
        'The sight of that festering wound makes my skin crawl.'
        'I\'m appalled by your lack of hygiene and basic cleanliness.',
        'The mere thought of their morally reprehensible actions sickens me.',

    ],
    [ # fearful
        'I\'m terrified of what might happen next.',
        'Please, don\'t leave me alone in the dark.',
        'I can\'t shake this feeling of impending doom.',
        'My heart is racing, and I\'m breaking out in a cold sweat.',
        'I\'m paralyzed with fear and can\'t move a muscle.',

    ],
    [ # happy
        'I\'m over the moon with joy right now!',
        'This is the best news I\'ve heard in ages.',
        'I can\'t stop smiling; today has been perfect.',
        'I\'m so glad we got to spend this time together.',
        'I feel like I\'m on top of the world!',
    ],
    [ # neutral
        'The weather today is pretty average.',
        'I need to pick up some groceries after work.',
        'The meeting is scheduled for 2 PM tomorrow.',
        'I think I\'ll have a sandwich for lunch.',
        'The bus should be here in about five minutes.',
    ],
    [ # other
        'I\'m not quite sure how to feel about this situation.',
        'This is unlike anything I\'ve ever experienced before.',
        'I\'m experiencing a mix of emotions right now.',
        'I can\'t put my finger on what I\'m feeling at the moment.',
        'There\'s something strange and unfamiliar about this.',
    ],
    [ # sad
        'I feel like my heart has been shattered into a million pieces.',
        'I can\'t stop crying; the pain is just too much to bear.',
        'I\'ve never felt so alone and hopeless in my life.',
        'It\'s like there\'s a heavy weight on my chest, and I can\'t breathe.',
        'I don\'t know how I\'ll ever recover from this loss.',
    ],
    [ # surprised
        'Wow, I never saw that coming!',
        'I\'m utterly shocked by this unexpected turn of events.',
        'I can\'t believe my eyes; this is incredible!',
        'I\'m at a complete loss for words right now.',
        'I never thought I\'d see the day when this would happen.',
    ],
    [ # unknown
        'I\'m not sure what to make of this situation.',
        'There\'s an air of mystery surrounding these events.',
        'I can\'t quite put my finger on what\'s happening here.',
        'The truth behind this remains elusive and unclear.',
        'I\'m left with more questions than answers at this point.',
    ],
]

IDX_TO_EMOTION = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'other',
    6: 'sad',
    7: 'surprised',
    8: 'unknown',
}

EMOTION_TO_IDX = {
    'angry': 0,
    'disgusted': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'other': 5,
    'sad': 6,
    'surprised': 7,
    'unknown': 8,
}
