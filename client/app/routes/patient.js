import Route from '@ember/routing/route';
import axios from 'axios';
import { inject as service } from '@ember/service';
import AuthenticatedRoute from './authenticated-route';

export default class PatientRoute extends AuthenticatedRoute {
  @service ecgData;

  async model(params) {
    this.ecgData.isrecordFetched = false;
    console.log(params);
    console.log('from model in paitent');
    const data = await this.ecgData.getRecordList(params.id);
    // console.log(data);
    return data;
    // return params.id;
  }
}
