'''
release history
'''

import collections


# release history
Tag = collections.namedtuple('Tag', 'version date')
release_history = (
    Tag('0.5.0', '2023-08-25'),
    Tag('0.4.1', '2023-08-14'),
    Tag('0.4.0', '2023-07-26'),
    Tag('0.3.1', '2023-06-01'),
    Tag('0.3.0', '2023-05-31'),
    Tag('0.1.5', '2023-05-10'),
    Tag('0.1.4', '2023-03-23'),
    Tag('0.1.3', '2022-12-21'),
    Tag('0.1.2', '2022-07-21'),
    Tag('0.1.1', '2022-06-08'),
    Tag('0.1.0', '2022-06-07'),
)

# latest release version number and date
release_version = release_history[0].version
release_date = release_history[0].date
