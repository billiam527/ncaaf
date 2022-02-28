import pandas as pd


def transform_espn_ncaaf_team_data(json_data):

    """
    param json_data: (json) data pulled from espn's ncaaf api at the game level

    return df: (pd.DataFrame) data in the relational table form
    """

    # team data
    id = []
    slug = []
    location = []
    name = []
    nickname = []
    abbreviation = []
    display_name = []
    short_display_name = []

    for sport in json_data['sports']:
        for league in sport['leagues']:
            for team in league['teams']:
                id.append(team['team']['id'])
                slug.append(team['team']['slug'])
                location.append(team['team']['location'])
                name.append(team['team']['name'])
                nickname.append(team['team']['nickname'])
                abbreviation.append(team['team']['abbreviation'])
                display_name.append(team['team']['displayName'])
                short_display_name.append(team['team']['shortDisplayName'])

    columns = list(zip(id,
                       slug,
                       location,
                       name,
                       nickname,
                       abbreviation,
                       display_name,
                       short_display_name))

    col_names = ['id',
                 'slug',
                 'location',
                 'name',
                 'nickname',
                 'abbreviation',
                 'display_name',
                 'short_display_name']

    df = pd.DataFrame(columns, columns=col_names)

    df['id'] = df['id'].astype('int32')

    return df


if __name__ == '__main__':
    None
