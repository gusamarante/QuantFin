import requests
import pandas as pd


class IMF(object):

    # TODO Create Examples

    @staticmethod
    def dataflow():
        """
        Returns a pandas DataFrame where the first column 'Database ID' is the database ID for further usage.
        The second column 'Database Name' is the description of each database.
        """

        url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/Dataflow'
        raw_request = requests.get(url)
        df_request = pd.DataFrame(raw_request.json()['Structure']['Dataflows']['Dataflow'])

        ids = df_request['KeyFamilyRef'].apply(lambda x: x['KeyFamilyID'])
        names = df_request['Name'].apply(lambda x: x['#text'])

        return pd.DataFrame({'Database ID': ids, 'Database Name': names})

    def data_structure(self, databaseID, check_query=True):

        if check_query:
            available_datasets = self.dataflow()['Database ID'].tolist()

            if databaseID not in available_datasets:
                print('Database not available')
                return None

        url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/DataStructure/' + databaseID
        raw_request = requests.get(url)

        if not raw_request.ok:
            print('Request returned nothing')
            return None

        rparsed = raw_request.json()['Structure']
        dim_code = pd.Series(rparsed['KeyFamilies']['KeyFamily']['Components']['Dimension']).apply(
            lambda x: x['@codelist']).tolist()
        dim_codedict = [x for x in rparsed['CodeLists']['CodeList'] if x['@id'] in dim_code]
        dim_codedict = dict(zip(pd.Series(dim_codedict).apply(lambda x: x['@id']).tolist(), dim_codedict))

        for k in dim_codedict.keys():
            dim_codedict[k] = pd.DataFrame({'CodeValue': pd.Series(dim_codedict[k]['Code']).apply(lambda x: x['@value']),
                                            'CodeText': pd.Series(dim_codedict[k]['Code']).apply(lambda x: x['Description']['#text'])})

        return dim_code, dim_codedict

    def compact_data(self, database_id, queryfilter, series_name, startdate=None, enddate=None,
                     checkquery=False, verbose=False):
        """
        Returns a pandas DataFrame with all metadata
        """

        if checkquery:
            available_datasets = self.dataflow()['DatabaseID'].tolist()
            if database_id not in available_datasets:
                return None

        request_url = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/' + database_id + '/'

        for k in queryfilter.keys():
            request_url = request_url + queryfilter[k] + '.'

        if not (startdate is None):
            request_url = request_url + '?startPeriod=' + startdate

        if not (enddate is None):
            request_url = request_url + '?endPeriod=' + enddate

        raw_request = requests.get(request_url)

        if verbose:
            print('\nmaking API call:\n')
            print(request_url)
            print('\n')

        if not raw_request.ok:
            print('Bad request')
            return None

        rparsed = raw_request.json()

        if 'Series' not in list(rparsed['CompactData']['DataSet'].keys()):
            print('no data available \n')
            return None

        return_df = pd.DataFrame(rparsed['CompactData']['DataSet']['Series']['Obs']).set_index('@TIME_PERIOD')
        return_df.index.rename(None, inplace=True)
        return_df = return_df[['@OBS_VALUE']]
        return_df.columns = [series_name]
        return_df[series_name] = return_df[series_name].astype(float)

        return return_df
