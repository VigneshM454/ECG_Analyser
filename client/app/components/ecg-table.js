import Component from '@glimmer/component';
import { inject as service } from '@ember/service';
export default class EcgTableComponent extends Component {
  @service ecgData;

  constructor() {
    super(...arguments);
    this.ecgData.getPatientList();
  }
}
