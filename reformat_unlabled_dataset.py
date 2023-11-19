import pandas as pd
from sklearn.preprocessing import LabelEncoder


def reformat_unlabled_dataset():
    csv_path = ('C:\\Users\\Gubay\\OneDrive\\Documents\\Archive_University\\Мага_3\\ml_course_work\\datasets'
                    '\\kdd_50000_unlabeled.csv')
    data = pd.read_csv(csv_path)

    data_encoded = data.copy(deep=True)
    label_encoder = LabelEncoder()
    data_encoded['protocol_type_encoded'] = label_encoder.fit_transform(data_encoded['protocol_type'])
    data_encoded['service_encoded'] = label_encoder.fit_transform(data_encoded['service'])
    data_encoded['flag_encoded'] = label_encoder.fit_transform(data_encoded['flag'])

    data_encoded = data_encoded.drop('protocol_type', axis=1)
    data_encoded = data_encoded.drop('service', axis=1)
    data_encoded = data_encoded.drop('flag', axis=1)

    data = data_encoded.copy(deep=True)

    result = pd.DataFrame({
        'protocol_type_encoded': data['protocol_type_encoded'],
        'service_encoded': data['service_encoded'],
        'logged_in': data["logged_in"],
        'count': data["count"],
        'srv_count': data["srv_count"],
        'dst_host_count': data['dst_host_count'],
        'temp0': 0,
        'dst_host_same_src_port_rate': data['dst_host_same_src_port_rate']
    })

    result.to_csv('datasets\\kdd_50000_unlabled_modified.csv', index=False)

    print('data transformation ended')


if __name__ == "__main__":
    reformat_unlabled_dataset()
