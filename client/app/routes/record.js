import Route from '@ember/routing/route';
import {inject as service} from '@ember/service'
import AuthenticatedRoute from './authenticated-route';

export default class RecordRoute extends AuthenticatedRoute {
  @service ecgData;

  async model(params) {
    console.log(params);
    console.log('from model in record');
    const data =  await this.ecgData.getUniqueRecord(params.id);
    console.log(data);
    // console.log(data);
    return data.record?.result
    // return params.id;
  }
}
