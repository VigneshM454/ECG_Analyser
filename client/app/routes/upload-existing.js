import Route from '@ember/routing/route';
import AuthenticatedRoute from './authenticated-route';

export default class UploadExistingRoute extends AuthenticatedRoute {
  model(params) {
    console.log(params);
    return params.id;
  }
}
